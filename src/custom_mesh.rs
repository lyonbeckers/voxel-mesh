use crate::node;
use gdnative::api::{GeometryInstance, ImmediateGeometry, Mesh};
use gdnative::prelude::*;
use legion::*;
use std::collections::HashMap;

pub struct MeshData {
	pub verts: Vec<Vector3>,
	pub uvs: Vec<Vector2>,
	pub uv2s: Vec<Vector2>,
	pub normals: Vec<Vector3>,
	pub indices: Vec<u16>,
	pub calculate_normals: bool,
}

impl MeshData {
	pub fn clear(&mut self) {
		self.verts.clear();
		self.uvs.clear();
		self.uv2s.clear();
		self.normals.clear();
		self.indices.clear();
	}
}

pub struct Material {
	name: Option<&'static str>,
}

impl Material {
	pub fn new() -> Self {
		Material { name: None }
	}

	pub fn from_str(s: &'static str) -> Self {
		Material { name: Some(s) }
	}
}

impl MeshData {
	pub fn new() -> Self {
		MeshData {
			verts: Vec::new(),
			uvs: Vec::new(),
			uv2s: Vec::new(),
			normals: Vec::new(),
			indices: Vec::new(),
			calculate_normals: false,
		}
	}
}

pub struct ManuallyChange;

pub struct RequiresManualChange;

pub fn create_tag_system(node: Ref<Node>) -> impl systems::Runnable {
	SystemBuilder::new("custom_mesh_system")
		.read_component::<Material>()
		.with_query(<Entity>::query().filter(
			!component::<node::NodeRef>() & component::<MeshData>(),
		))
		.build(move |commands, world, _, query| {
			query.for_each(world, |entity| {
				let immediate_geometry = ImmediateGeometry::new();

				let node = unsafe {
					node::add_node(
						&node.assume_safe(),
						immediate_geometry.upcast(),
					)
				};

				commands
					.add_component(*entity, node::NodeRef::new(node));
			})
		})
}

pub fn create_draw_system() -> impl systems::Runnable {
	SystemBuilder::new("custom_mesh_system")
        .read_component::<Material>()
        .with_query(
            <(Entity, Read<MeshData>, Read<node::NodeRef>)>::query().filter(
                (component::<RequiresManualChange>() & component::<ManuallyChange>())
                    | (!component::<RequiresManualChange>() & maybe_changed::<MeshData>()),
            ),
        )
        .build(move |commands, world, _, query| {
            let mut entities: HashMap<Entity, Ref<ImmediateGeometry>> = HashMap::new();

            query.for_each(world, |(entity, mesh_data, node_ref)| {
                godot_print!("Drawing {:?}", unsafe {
                    node_ref.val().assume_safe().name()
                });

                let verts = &mesh_data.verts;
                let uvs = &mesh_data.uvs;
                let uv2s = &mesh_data.uv2s;
                let indices = &mesh_data.indices;

                let normals = if mesh_data.calculate_normals {
                    calculate_normals(&indices, verts)
                } else {
                    mesh_data.normals.clone()
                };

                let immediate_geometry: Ref<ImmediateGeometry> = unsafe {
                    node_ref
                        .val()
                        .assume_safe()
                        .cast::<ImmediateGeometry>()
                        .unwrap()
                        .assume_shared()
                };

                entities.insert(*entity, immediate_geometry);

                unsafe {
                    let immediate_geometry = immediate_geometry.assume_safe();

                    immediate_geometry.clear();
                    immediate_geometry.begin(Mesh::PRIMITIVE_TRIANGLES, Null::null());

                    let uv2s_len = uv2s.len();

                    for index in indices {
                        let index = *index as usize;

                        immediate_geometry.set_normal(normals[index]);
                        immediate_geometry.set_uv(uvs[index]);
                        if index < uv2s_len {
                            immediate_geometry.set_uv2(uv2s[index]);
                        }
                        immediate_geometry.add_vertex(verts[index]);
                    }

                    immediate_geometry.end();
                }
            });

            for (entity, immediate_geometry) in entities {
                commands.exec_mut(move |world, _| {
                    if let Some(mut entry) = world.entry(entity) {
                        if let Ok(material) = entry.get_component::<Material>() {
                            let resource = ResourceLoader::godot_singleton().load(
                                match material.name {
                                    Some(r) => r,
                                    None => {
                                        //TODO: make it so it grabs a default material if no name value is set.
                                        panic!("Material name returned None");
                                    }
                                },
                                "Material",
                                false,
                            );

                            unsafe {
                                immediate_geometry
                                    .assume_safe()
                                    .upcast::<GeometryInstance>()
                                    .set_material_override(
                                        match resource {
                                            Some(r) => r,
                                            None => {
                                                //TODO: Same thing, gotta get a default material if none is found
                                                panic!(
                                                    "Resource {:?} does not exist",
                                                    material.name
                                                );
                                            }
                                        }
                                        .cast::<gdnative::api::Material>()
                                        .unwrap(),
                                    );
                            }
                        }

                        entry.remove_component::<ManuallyChange>();
                    }
                });
            }
        })
}

fn calculate_normals(
	indices: &Vec<u16>,
	verts: &Vec<Vector3>,
) -> Vec<Vector3> {
	let mut wnormals: Vec<Vec<Vector3>> =
		std::iter::repeat(Vec::new()).take(verts.len()).collect();

	// https://stackoverflow.com/questions/45477806/general-method-for-calculating-smooth-vertex-normals-with-100-smoothness
	for i in (0..indices.len()).step_by(3) {
		let i = i as usize;

		let i1 = indices[i] as usize;
		let i2 = indices[i + 1] as usize;
		let i3 = indices[i + 2] as usize;

		let p1 = verts[i1];
		let p2 = verts[i2];
		let p3 = verts[i3];

		let n = (p2 - p1).cross(p3 - p1).abs();

		let a1 =
			((p2 - p1).normalized()).angle_to((p3 - p1).normalized());
		let a2 =
			((p3 - p2).normalized()).angle_to((p1 - p2).normalized());
		let a3 =
			((p1 - p3).normalized()).angle_to((p2 - p3).normalized());

		wnormals.get_mut(i1).unwrap().push(n * a1);
		wnormals.get_mut(i2).unwrap().push(n * a2);
		wnormals.get_mut(i3).unwrap().push(n * a3);
	}

	wnormals
		.into_iter()
		.map(|normals| {
			let mut final_normal = Vector3::ZERO;
			normals.iter().for_each(|normal| final_normal += *normal);
			final_normal.normalized()
		})
		.collect::<Vec<Vector3>>()
}
