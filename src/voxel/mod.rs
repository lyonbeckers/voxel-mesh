pub mod mesh;
pub mod tile_data;

use std::collections::{HashMap, HashSet};

use crate::{custom_mesh::RequiresManualChange, Aabb, Octree, Point};

use crate::custom_mesh;
use legion::*;
use octree::PointData;
use tile_data::TileData;

use crate::custom_mesh::MeshData;

use self::mesh::MapMeshData;

pub const TILE_DIMENSIONS: TileDimensions = TileDimensions {
    x: 1.0,
    y: 1.0,
    z: 1.0,
};

///ChangeType stores the range of the changes so that we can determine whether or not adjacent MapChunks actually need to change, and
/// the range of the original change for making comparisons
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ChangeType {
    Direct(Aabb),
    Indirect(Aabb),
    Changed(Aabb),
}

///ManuallyChange tells the map chunks to update, and the Aabb gives us more information about which columns we will be updating so that we don't have to update all of them.
/// In most cases, we only need one Aabb, but we store it in a Vec for cases where two chunks that are separated by a chunk update simultaneously, effectively overwriting each other's
/// values. This means that ManuallyChange should be attempted to be got with get_component_mut in case it can be updated instead of being written as a new value.
#[derive(Clone, Debug, PartialEq)]
struct ManuallyChange {
    ranges: Vec<ChangeType>,
}

pub struct TileDimensions {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct CoordPos {
    pub value: Point,
}

impl CoordPos {
    pub fn new(point: Point) -> CoordPos {
        CoordPos { value: point }
    }
}

impl Default for CoordPos {
    fn default() -> CoordPos {
        CoordPos {
            value: Point::new(0, 0, 0),
        }
    }
}

/// Applies the const TILE_DIMENSIONS to each map coord to get its conversion in 3D space.
pub fn map_coords_to_world(map_coord: Point) -> nalgebra::Vector3<f32> {
    nalgebra::Vector3::<f32>::new(
        map_coord.x as f32 * TILE_DIMENSIONS.x,
        map_coord.y as f32 * TILE_DIMENSIONS.y,
        map_coord.z as f32 * TILE_DIMENSIONS.z,
    )
}

#[derive(Copy, Clone)]
pub struct Map {
    chunk_dimensions: Point,
}

impl Map {
    pub fn new(chunk_dimensions: Point) -> Self {
        Self { chunk_dimensions }
    }

    // Executes changes to the world map in octree. Takes an optional u32 as a client_id for store_history
    pub fn change(
        &self,
        world: &mut legion::world::World,
        octree: Octree,
        // store_history: Option<u32>,
    ) {
        // match self.can_change(world, &octree) {
        //     Err(_) => return,
        //     Ok((original_state, new_state)) => {
        //         if let Some(client_id) = store_history {
        //             let mut query = <(Write<History>, Read<ClientID>)>::query();

        //             if let Some((history, _)) =
        //                 query.iter_mut(world).find(|(_, id)| id.val() == client_id)
        //             {
        //                 history.add_step(StepType::MapChange((original_state, new_state)));
        //             }
        //         }
        //     }
        // }

        let mut entities: HashMap<Entity, MapChunkData> = HashMap::new();

        let aabb = octree.get_aabb();

        let mut map_chunk_exists_query = <(Entity, Read<MapChunkData>, Read<Point>)>::query();
        let existing_chunks = map_chunk_exists_query
            .iter(world)
            .map(|(e, m, p)| (*e, (*m).clone(), (*p).clone()))
            .collect::<Vec<(Entity, MapChunkData, Point)>>();
        let (existing, new): (Vec<_>, Vec<_>) = self
            .range_sliced_to_chunks(aabb)
            .into_iter()
            .partition(|(pt, _)| existing_chunks.iter().any(|(_, _, p)| pt == p));

        existing_chunks
            .into_iter()
            .filter(|(_, _, pt)| existing.iter().any(|(p, _)| pt == p))
            .for_each(|(entity, map_chunk, chunk_pt)| {
                tracing::debug!(target: "chunk exists", chunk_pt = ?chunk_pt);
                entities.insert(entity, map_chunk.clone());
            });

        let octrees = new
            .iter()
            .map(|(pt, _)| {
                Octree::new(
                    Aabb::new(
                        Point::new(
                            pt.x * self.chunk_dimensions.x + self.chunk_dimensions.x / 2,
                            pt.y * self.chunk_dimensions.y + self.chunk_dimensions.y / 2,
                            pt.z * self.chunk_dimensions.z + self.chunk_dimensions.z / 2,
                        ),
                        self.chunk_dimensions,
                    ),
                    octree::DEFAULT_MAX,
                )
            })
            .collect::<Vec<Octree>>();

        entities.extend(self.insert_mapchunks_with_octrees(&octrees, world, false));

        for (entity, map_data) in &mut entities {
            let map_aabb = map_data.octree.get_aabb();
            let overlap_aabb = aabb.get_intersection(map_aabb);

            let map_query_range = map_data.octree.query_range(overlap_aabb);
            let input_query_range = octree.query_range(overlap_aabb);

            let set = input_query_range.into_iter().collect::<HashSet<TileData>>();
            let map_set = map_query_range.into_iter().collect::<HashSet<TileData>>();

            if set.symmetric_difference(&map_set).count() == 0 {
                tracing::debug!("Set and map_set were symmetrically the same");
                continue;
            }

            //Remove any data that is in map_set but not set
            let difference = map_set.difference(&set);
            difference.into_iter().for_each(|item| {
                map_data.octree.remove_item(item.get_point());
            });

            //Add any data that is in set but not map_set
            let difference = set.difference(&map_set).copied().collect::<Vec<TileData>>();
            if !difference.is_empty() {
                map_data.octree.insert_elements(difference).unwrap();
            }

            // And the range of change to the ManuallyChange component if it exists, otherwise, make it exist
            if let Some(mut entry) = world.entry(*entity) {
                entry.add_component(map_data.clone());

                match entry.get_component_mut::<ManuallyChange>() {
                    Ok(change) => change.ranges.push(ChangeType::Direct(aabb)),
                    _ => entry.add_component(ManuallyChange {
                        ranges: vec![ChangeType::Direct(aabb)],
                    }),
                }
            }
        }
    }

    /// Returns AABBs that are subdivided to fit into the constraints of the chunk dimensions, as well as the chunk pt they'd fit in
    pub fn range_sliced_to_chunks(&self, aabb: Aabb) -> Vec<(Point, Aabb)> {
        let min = aabb.get_min();
        let max = aabb.get_max();

        let x_min_chunk = (min.x as f32 / self.chunk_dimensions.x as f32).floor() as i32;
        let y_min_chunk = (min.y as f32 / self.chunk_dimensions.y as f32).floor() as i32;
        let z_min_chunk = (min.z as f32 / self.chunk_dimensions.z as f32).floor() as i32;

        let x_max_chunk = (max.x as f32 / self.chunk_dimensions.x as f32).floor() as i32 + 1;
        let y_max_chunk = (max.y as f32 / self.chunk_dimensions.y as f32).floor() as i32 + 1;
        let z_max_chunk = (max.z as f32 / self.chunk_dimensions.z as f32).floor() as i32 + 1;

        let min_chunk = Point::new(x_min_chunk, y_min_chunk, z_min_chunk);

        let dimensions = Point::new(x_max_chunk, y_max_chunk, z_max_chunk) - min_chunk;

        let volume = dimensions.x * dimensions.y * dimensions.z;

        let mut results = Vec::new();

        for i in 0..volume {
            let x = x_min_chunk + i % dimensions.x;
            let y = y_min_chunk + (i / dimensions.x) % dimensions.y;
            let z = z_min_chunk + i / (dimensions.x * dimensions.y);

            let min = Point::new(
                x * self.chunk_dimensions.x,
                y * self.chunk_dimensions.y,
                z * self.chunk_dimensions.z,
            );
            let max = min + self.chunk_dimensions;

            results.push((
                Point::new(x, y, z),
                Aabb::from_extents(min, max).get_intersection(aabb),
            ));
        }

        results
    }

    /// Does a query range on every chunk that fits within the range
    pub fn query_chunk_range<T: IntoIterator<Item = (Entity, MapChunkData, Point)> + Clone>(
        &self,
        map_datas: T,
        range: Aabb,
    ) -> Vec<TileData> {
        let mut results = Vec::new();

        self.chunks_in_range(map_datas, range)
            .iter()
            .for_each(|(_, map_data)| results.extend(map_data.octree.query_range(range)));

        results
    }

    pub fn chunks_in_range<T: IntoIterator<Item = (Entity, MapChunkData, Point)> + Clone>(
        &self,
        map_datas: T,
        range: Aabb,
    ) -> Vec<(Entity, MapChunkData)> {
        let min = range.get_min();
        let max = range.get_max();

        let x_min_chunk = (min.x as f32 / self.chunk_dimensions.x as f32).floor() as i32;
        let y_min_chunk = (min.y as f32 / self.chunk_dimensions.y as f32).floor() as i32;
        let z_min_chunk = (min.z as f32 / self.chunk_dimensions.z as f32).floor() as i32;

        let x_max_chunk = (max.x as f32 / self.chunk_dimensions.x as f32).floor() as i32 + 1;
        let y_max_chunk = (max.y as f32 / self.chunk_dimensions.y as f32).floor() as i32 + 1;
        let z_max_chunk = (max.z as f32 / self.chunk_dimensions.z as f32).floor() as i32 + 1;

        let min_chunk = Point::new(x_min_chunk, y_min_chunk, z_min_chunk);

        let dimensions = Point::new(x_max_chunk, y_max_chunk, z_max_chunk) - min_chunk;

        let volume = dimensions.x * dimensions.y * dimensions.z;

        let mut results: Vec<(Entity, MapChunkData)> = Vec::new();

        for i in 0..volume {
            let x = x_min_chunk + i % dimensions.x;
            let y = y_min_chunk + (i / dimensions.x) % dimensions.y;
            let z = z_min_chunk + i / (dimensions.x * dimensions.y);

            let point = Point::new(x, y, z);

            map_datas
                .clone()
                .into_iter()
                .filter(|(_, _, pt)| *pt == point)
                .for_each(|(entity, map_data, _)| {
                    results.push((entity, map_data));
                });
        }

        results
    }

    pub fn insert_mapchunks_with_octrees(
        self,
        octrees: &[Octree],
        world: &mut World,
        changed: bool,
    ) -> Vec<(Entity, MapChunkData)> {
        if changed {
            let (components, map_chunks): (Vec<_>, Vec<_>) = octrees
                .into_iter()
                .map(|octree| {
                    let map_data = MapChunkData {
                        octree: octree.clone(),
                    };
                    let chunk_pt = map_data.get_chunk_point();
                    let area = self.chunk_dimensions.x * self.chunk_dimensions.z;

                    (
                        (
                            ManuallyChange {
                                ranges: vec![ChangeType::Direct(octree.get_aabb())],
                            },
                            chunk_pt,
                            map_data.clone(),
                            MeshData::new(),
                            MapMeshData::new(
                                (0..area).map(|_| mesh::VertexData::default()).collect(),
                            ),
                            RequiresManualChange {},
                        ),
                        map_data,
                    )
                })
                .unzip();

            let entities = world.extend(components);
            entities
                .into_iter()
                .copied()
                .zip(map_chunks.into_iter())
                .collect()
        } else {
            let (components, map_chunks): (Vec<_>, Vec<_>) = octrees
                .into_iter()
                .map(|octree| {
                    let map_data = MapChunkData {
                        octree: octree.clone(),
                    };
                    let chunk_pt = map_data.get_chunk_point();
                    let area = self.chunk_dimensions.x * self.chunk_dimensions.z;

                    (
                        (
                            chunk_pt,
                            map_data.clone(),
                            MeshData::new(),
                            mesh::MapMeshData::new(
                                (0..area).map(|_| mesh::VertexData::default()).collect(),
                            ),
                            custom_mesh::RequiresManualChange {},
                        ),
                        map_data,
                    )
                })
                .unzip();

            let entities = world.extend(components);
            entities
                .into_iter()
                .copied()
                .zip(map_chunks.into_iter())
                .collect()
        }
    }

    /// Inserts a new mapchunk with the octree data into world. Prefer to use
    /// insert_mapchunks_with_octrees if you can
    pub fn insert_mapchunk_with_octree(
        self,
        octree: &Octree,
        world: &mut World,
        changed: bool,
    ) -> (Entity, MapChunkData) {
        let map_data = MapChunkData {
            octree: octree.clone(),
        };

        let chunk_pt = map_data.get_chunk_point();

        let area = self.chunk_dimensions.x * self.chunk_dimensions.z;

        if changed {
            (
                world.push((
                    ManuallyChange {
                        ranges: vec![ChangeType::Direct(octree.get_aabb())],
                    },
                    chunk_pt,
                    map_data.clone(),
                    MeshData::new(),
                    mesh::MapMeshData::new(
                        (0..area).map(|_| mesh::VertexData::default()).collect(),
                    ),
                    custom_mesh::RequiresManualChange {},
                )),
                map_data,
            )
        } else {
            (
                world.push((
                    chunk_pt,
                    map_data.clone(),
                    MeshData::new(),
                    mesh::MapMeshData::new(
                        (0..area).map(|_| mesh::VertexData::default()).collect(),
                    ),
                    custom_mesh::RequiresManualChange {},
                )),
                map_data,
            )
        }
    }
}

impl Default for Map {
    fn default() -> Self {
        Map {
            chunk_dimensions: Point::new(10, 10, 10),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapChunkData {
    pub octree: Octree,
}

impl MapChunkData {
    pub fn new(aabb: Aabb) -> Self {
        MapChunkData {
            octree: Octree::new(aabb, octree::DEFAULT_MAX),
        }
    }

    pub fn get_chunk_point(&self) -> Point {
        let aabb = self.octree.get_aabb();
        let min = aabb.get_min();
        let dimensions = aabb.dimensions;

        Point::new(
            (min.x as f32 / dimensions.x as f32).floor() as i32,
            (min.y as f32 / dimensions.y as f32).floor() as i32,
            (min.z as f32 / dimensions.z as f32).floor() as i32,
        )
    }
}
