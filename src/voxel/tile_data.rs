use octree::PointData;

use crate::Point;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct TileData {
    point: Point,
    tile: u32,
}

impl TileData {
    pub fn new(point: Point, tile: u32) -> Self {
        Self { point, tile }
    }

    pub fn get_tile(&self) -> u32 {
        self.tile
    }
}

impl PointData<i32> for TileData {
    fn get_point(&self) -> Point {
        self.point
    }
}
