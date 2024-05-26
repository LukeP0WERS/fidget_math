use std::collections::HashMap;

struct ContextBuilder(tera::Context);

impl ContextBuilder {
    pub fn new() -> Self {
        Self(tera::Context::new())
    }

    pub fn new_vecn(dim: u32) -> Self {
        ContextBuilder::new()
            .with_template("vec.rs.tera")
            .with_dimension(dim)
            .with_is_align(false)
    }

    pub fn new_tree_float() -> Self {
        Self::new_vecn(1).with_tree().with_is_float()
    }

    pub fn new_vec2() -> Self {
        Self::new_vecn(2).with_tree()
    }

    pub fn new_vec3() -> Self {
        Self::new_vecn(3).with_tree()
    }

    pub fn new_vec4() -> Self {
        Self::new_vecn(4).with_tree().with_is_align(true)
    }

    pub fn with_template(mut self, template_path: &str) -> Self {
        self.0.insert("template_path", template_path);
        self
    }

    pub fn with_tree(mut self) -> Self {
        self.0.insert("scalar_t", "Tree");
        self
    }

    pub fn with_is_float(mut self) -> Self {
        self.0.insert("is_float", &true);
        self
    }

    fn with_dimension(mut self, dim: u32) -> Self {
        self.0.insert("dim", &dim);
        self
    }

    fn with_is_align(mut self, is_align: bool) -> Self {
        self.0.insert("is_align", &is_align);
        self
    }

    pub fn build(self) -> tera::Context {
        self.0
    }
}

pub fn build_output_pairs() -> HashMap<&'static str, tera::Context> {
    HashMap::from([
        ("src/tree/tree_float.rs", ContextBuilder::new_tree_float().build()),
        ("src/tree/vec2.rs", ContextBuilder::new_vec2().build()),
        ("src/tree/vec3.rs", ContextBuilder::new_vec3().build()),
        ("src/tree/vec4.rs", ContextBuilder::new_vec4().build()),
    ])
}
