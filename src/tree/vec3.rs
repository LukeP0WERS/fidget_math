// Generated from vec.rs.tera template. Edit the template, not the generated file.

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::*;
use fidget::context::Tree;

use super::vec2::Vec2;

use super::vec4::Vec4;

/// Creates a 3-dimensional vector.
#[inline(always)]
#[must_use]
pub fn vec3(x: Tree, y: Tree, z: Tree) -> Vec3 {
    Vec3::new(x, y, z)
}

/// A 3-dimensional vector.
#[derive(Clone, PartialEq)]
pub struct Vec3 {
    pub x: Tree,
    pub y: Tree,
    pub z: Tree,
}

impl Vec3 {
    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub fn new(x: impl Into<Tree>, y: impl Into<Tree>, z: impl Into<Tree>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    #[must_use]
    pub fn splat(v: impl Into<Tree> + Clone) -> Self {
        Self {
            x: v.clone().into(),

            y: v.clone().into(),

            z: v.clone().into(),
        }
    }

    /// Creates a new vector from an array.
    #[inline]
    #[must_use]
    pub fn from_array(a: [Tree; 3]) -> Self {
        Self::new(a[0].clone(), a[1].clone(), a[2].clone())
    }

    /// `[x, y, z]`
    #[inline]
    #[must_use]
    pub fn to_array(&self) -> [Tree; 3] {
        [self.x.clone(), self.y.clone(), self.z.clone()]
    }

    /// Creates a vector from the first 3 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 3 elements long.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[Tree]) -> Self {
        Self::new(slice[0].clone(), slice[1].clone(), slice[2].clone())
    }

    /// Writes the elements of `self` to the first 3 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 3 elements long.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [Tree]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
    }

    /// Internal method for creating a 3D vector from a 4D vector, discarding `w`.
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub(crate) fn from_vec4(v: Vec4) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    /// Creates a 4D vector from `self` and the given `w` value.
    #[inline]
    #[must_use]
    pub fn extend(self, w: Tree) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    /// Creates a 2D vector from the `x` and `y` elements of `self`, discarding `z`.
    #[inline]
    #[must_use]
    pub fn truncate(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Creates a 3D vector from `self` with the given value of `x`.
    #[inline]
    #[must_use]
    pub fn with_x(mut self, x: Tree) -> Self {
        self.x = x;
        self
    }

    /// Creates a 3D vector from `self` with the given value of `y`.
    #[inline]
    #[must_use]
    pub fn with_y(mut self, y: Tree) -> Self {
        self.y = y;
        self
    }

    /// Creates a 3D vector from `self` with the given value of `z`.
    #[inline]
    #[must_use]
    pub fn with_z(mut self, z: Tree) -> Self {
        self.z = z;
        self
    }

    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> Tree {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    /// Returns a vector where every component is the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self::splat(self.dot(rhs))
    }

    /// Computes the cross product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y.clone() * rhs.z.clone() - rhs.y.clone() * self.z.clone(),
            y: self.z.clone() * rhs.x.clone() - rhs.z.clone() * self.x.clone(),
            z: self.x.clone() * rhs.y.clone() - rhs.x.clone() * self.y.clone(),
        }
    }

    /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[self.x.min(rhs.x), self.y.min(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
            z: self.z.min(rhs.z),
        }
    }

    /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[self.x.max(rhs.x), self.y.max(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
        }
    }

    /// Component-wise clamping of values, similar to [`Tree::clamp`].
    ///
    /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
    #[inline]
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns the horizontal minimum of `self`.
    ///
    /// In other words this computes `min(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn min_element(self) -> Tree {
        self.x.min(self.y.min(self.z))
    }

    /// Returns the horizontal maximum of `self`.
    ///
    /// In other words this computes `max(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn max_element(self) -> Tree {
        self.x.max(self.y.max(self.z))
    }

    /// Returns the sum of all elements of `self`.
    ///
    /// In other words, this computes `self.x + self.y + ..`.
    #[inline]
    #[must_use]
    pub fn element_sum(self) -> Tree {
        self.x + self.y + self.z
    }

    /// Returns the product of all elements of `self`.
    ///
    /// In other words, this computes `self.x * self.y * ..`.
    #[inline]
    #[must_use]
    pub fn element_product(self) -> Tree {
        self.x * self.y * self.z
    }

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(self) -> Tree {
        self.clone().dot(self).sqrt()
    }

    /// Computes the squared length of `self`.
    ///
    /// This is faster than `length()` as it avoids a square root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> Tree {
        self.clone().dot(self)
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(self) -> Tree {
        Tree::constant(1.0) / self.length()
    }

    /// Computes the Euclidean distance between two points in space.
    #[inline]
    #[must_use]
    pub fn distance(self, rhs: Self) -> Tree {
        (self - rhs).length()
    }

    /// Returns `self` normalized to length 1.0.
    ///
    /// For valid results, `self` must _not_ be of length zero, nor very close to zero.
    ///
    /// See also [`Self::try_normalize()`] and [`Self::normalize_or_zero()`].
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        #[allow(clippy::let_and_return)]
        let normalized = self.clone().mul(self.length_recip());
        normalized
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be of non-zero length.
    #[inline]
    #[must_use]
    pub fn project_onto(self, rhs: Self) -> Self {
        let other_len_sq_rcp = Tree::constant(1.0) / rhs.clone().dot(rhs.clone());
        rhs.clone() * self.dot(rhs) * other_len_sq_rcp
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be of non-zero length.
    #[inline]
    #[must_use]
    pub fn reject_from(self, rhs: Self) -> Self {
        self.clone() - self.project_onto(rhs)
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be normalized.
    #[inline]
    #[must_use]
    pub fn project_onto_normalized(self, rhs: Self) -> Self {
        rhs.clone() * self.dot(rhs)
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be normalized.
    #[inline]
    #[must_use]
    pub fn reject_from_normalized(self, rhs: Self) -> Self {
        self.clone() - self.project_onto_normalized(rhs)
    }

    /// Returns a vector containing the nearest integer to a number for each element of `self`.
    /// Round half-way cases away from 0.0.
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    /// Returns a vector containing the largest integer less than or equal to a number for each
    /// element of `self`.
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Returns a vector containing the smallest integer greater than or equal to a number for
    /// each element of `self`.
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /*
    /// Returns a vector containing the integer part each element of `self`. This means numbers are
    /// always truncated towards zero.
    #[inline]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self {

                x: self.x.trunc(),
                y: self.y.trunc(),
                z: self.z.trunc(),
        }
    }
    */

    /*
    /// Returns a vector containing the fractional part of the vector as `self - self.trunc()`.
    ///
    /// Note that this differs from the GLSL implementation of `fract` which returns
    /// `self - self.floor()`.
    ///
    /// Note that this is fast but not precise for large numbers.
    #[inline]
    #[must_use]
    pub fn fract(self) -> Self {
        self - self.trunc()
    }
    */

    /// Returns a vector containing the fractional part of the vector as `self - self.floor()`.
    ///
    /// Note that this differs from the Rust implementation of `fract` which returns
    /// `self - self.trunc()`.
    ///
    /// Note that this is fast but not precise for large numbers.
    #[inline]
    #[must_use]
    pub fn fract_gl(self) -> Self {
        self.clone() - self.floor()
    }

    /// Returns a vector containing `e^self` (the exponential function) for each element of
    /// `self`.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        Self::new(self.x.exp(), self.y.exp(), self.z.exp())
    }

    /*
    /// Returns a vector containing each element of `self` raised to the power of `n`.
    #[inline]
    #[must_use]
    pub fn powf(self, n: Tree) -> Self {
        Self::new(

                math::powf(self.x, n),
                math::powf(self.y, n),
                math::powf(self.z, n),
        )
    }
    */

    /// Returns a vector containing the reciprocal `1.0/n` of each element of `self`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self {
            x: Tree::constant(1.0) / self.x,
            y: Tree::constant(1.0) / self.y,
            z: Tree::constant(1.0) / self.z,
        }
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[doc(alias = "mix")]
    #[inline]
    #[must_use]
    pub fn lerp(self, rhs: Self, s: Tree) -> Self {
        self.clone() + ((rhs - self) * s)
    }

    /// Calculates the midpoint between `self` and `rhs`.
    ///
    /// The midpoint is the average of, or halfway point between, two vectors.
    /// `a.midpoint(b)` should yield the same result as `a.lerp(b, 0.5)`
    /// while being slightly cheaper to compute.
    #[inline]
    pub fn midpoint(self, rhs: Self) -> Self {
        (self + rhs) * Tree::constant(0.5)
    }
}

impl Default for Vec3 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(Tree::constant(0.0))
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x.div(rhs.x),
            y: self.y.div(rhs.y),
            z: self.z.div(rhs.z),
        }
    }
}

impl DivAssign<Vec3> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x.div_assign(rhs.x);
        self.y.div_assign(rhs.y);
        self.z.div_assign(rhs.z);
    }
}

impl Div<Tree> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Tree) -> Self {
        Self {
            x: self.x.div(rhs.clone()),
            y: self.y.div(rhs.clone()),
            z: self.z.div(rhs.clone()),
        }
    }
}

impl DivAssign<Tree> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: Tree) {
        self.x.div_assign(rhs.clone());
        self.y.div_assign(rhs.clone());
        self.z.div_assign(rhs.clone());
    }
}

impl Div<Vec3> for Tree {
    type Output = Vec3;
    #[inline]
    fn div(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.clone().div(rhs.x),
            y: self.clone().div(rhs.y),
            z: self.clone().div(rhs.z),
        }
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x.mul(rhs.x),
            y: self.y.mul(rhs.y),
            z: self.z.mul(rhs.z),
        }
    }
}

impl MulAssign<Vec3> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x.mul_assign(rhs.x);
        self.y.mul_assign(rhs.y);
        self.z.mul_assign(rhs.z);
    }
}

impl Mul<Tree> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Tree) -> Self {
        Self {
            x: self.x.mul(rhs.clone()),
            y: self.y.mul(rhs.clone()),
            z: self.z.mul(rhs.clone()),
        }
    }
}

impl MulAssign<Tree> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Tree) {
        self.x.mul_assign(rhs.clone());
        self.y.mul_assign(rhs.clone());
        self.z.mul_assign(rhs.clone());
    }
}

impl Mul<Vec3> for Tree {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.clone().mul(rhs.x),
            y: self.clone().mul(rhs.y),
            z: self.clone().mul(rhs.z),
        }
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x.add(rhs.x),
            y: self.y.add(rhs.y),
            z: self.z.add(rhs.z),
        }
    }
}

impl AddAssign<Vec3> for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x.add_assign(rhs.x);
        self.y.add_assign(rhs.y);
        self.z.add_assign(rhs.z);
    }
}

impl Add<Tree> for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Tree) -> Self {
        Self {
            x: self.x.add(rhs.clone()),
            y: self.y.add(rhs.clone()),
            z: self.z.add(rhs.clone()),
        }
    }
}

impl AddAssign<Tree> for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Tree) {
        self.x.add_assign(rhs.clone());
        self.y.add_assign(rhs.clone());
        self.z.add_assign(rhs.clone());
    }
}

impl Add<Vec3> for Tree {
    type Output = Vec3;
    #[inline]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.clone().add(rhs.x),
            y: self.clone().add(rhs.y),
            z: self.clone().add(rhs.z),
        }
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x.sub(rhs.x),
            y: self.y.sub(rhs.y),
            z: self.z.sub(rhs.z),
        }
    }
}

impl SubAssign<Vec3> for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec3) {
        self.x.sub_assign(rhs.x);
        self.y.sub_assign(rhs.y);
        self.z.sub_assign(rhs.z);
    }
}

impl Sub<Tree> for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Tree) -> Self {
        Self {
            x: self.x.sub(rhs.clone()),
            y: self.y.sub(rhs.clone()),
            z: self.z.sub(rhs.clone()),
        }
    }
}

impl SubAssign<Tree> for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Tree) {
        self.x.sub_assign(rhs.clone());
        self.y.sub_assign(rhs.clone());
        self.z.sub_assign(rhs.clone());
    }
}

impl Sub<Vec3> for Tree {
    type Output = Vec3;
    #[inline]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.clone().sub(rhs.x),
            y: self.clone().sub(rhs.y),
            z: self.clone().sub(rhs.z),
        }
    }
}

impl Rem<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self {
            x: self.x.modulo(rhs.x),
            y: self.y.modulo(rhs.y),
            z: self.z.modulo(rhs.z),
        }
    }
}

impl RemAssign<Vec3> for Vec3 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.x = self.x.modulo(rhs.x);
        self.y = self.y.modulo(rhs.y);
        self.z = self.z.modulo(rhs.z);
    }
}

impl Rem<Tree> for Vec3 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Tree) -> Self {
        Self {
            x: self.x.modulo(rhs.clone()),
            y: self.y.modulo(rhs.clone()),
            z: self.z.modulo(rhs.clone()),
        }
    }
}

impl RemAssign<Tree> for Vec3 {
    #[inline]
    fn rem_assign(&mut self, rhs: Tree) {
        self.x = self.x.modulo(rhs.clone());
        self.y = self.y.modulo(rhs.clone());
        self.z = self.z.modulo(rhs.clone());
    }
}

impl Rem<Vec3> for Tree {
    type Output = Vec3;
    #[inline]
    fn rem(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.clone().modulo(rhs.x),
            y: self.clone().modulo(rhs.y),
            z: self.clone().modulo(rhs.z),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[Tree; 3]> for Vec3 {
    #[inline]
    fn as_ref(&self) -> &[Tree; 3] {
        unsafe { &*(self as *const Vec3 as *const [Tree; 3]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[Tree; 3]> for Vec3 {
    #[inline]
    fn as_mut(&mut self) -> &mut [Tree; 3] {
        unsafe { &mut *(self as *mut Vec3 as *mut [Tree; 3]) }
    }
}

impl Sum for Vec3 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::splat(Tree::constant(0.0)), Self::add)
    }
}

impl<'a> Sum<&'a Self> for Vec3 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::splat(Tree::constant(0.0)), |a, b| {
            Self::add(a, b.clone())
        })
    }
}

impl Product for Vec3 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::splat(Tree::constant(1.0)), Self::mul)
    }
}

impl<'a> Product<&'a Self> for Vec3 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::splat(Tree::constant(1.0)), |a, b| {
            Self::mul(a, b.clone())
        })
    }
}

impl Index<usize> for Vec3 {
    type Output = Tree;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for Vec3 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple(stringify!(Vec3))
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .finish()
    }
}

impl From<[Tree; 3]> for Vec3 {
    #[inline]
    fn from(a: [Tree; 3]) -> Self {
        Self::new(a[0].clone(), a[1].clone(), a[2].clone())
    }
}

impl From<Vec3> for [Tree; 3] {
    #[inline]
    fn from(v: Vec3) -> Self {
        [v.x, v.y, v.z]
    }
}

impl From<(Tree, Tree, Tree)> for Vec3 {
    #[inline]
    fn from(t: (Tree, Tree, Tree)) -> Self {
        Self::new(t.0, t.1, t.2)
    }
}

impl From<Vec3> for (Tree, Tree, Tree) {
    #[inline]
    fn from(v: Vec3) -> Self {
        (v.x, v.y, v.z)
    }
}

impl From<(Vec2, Tree)> for Vec3 {
    #[inline]
    fn from((v, z): (Vec2, Tree)) -> Self {
        Self::new(v.x, v.y, z)
    }
}
