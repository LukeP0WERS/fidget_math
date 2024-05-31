// Generated from vec.rs.tera template. Edit the template, not the generated file.

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::*;
use fidget::context::Tree;

use super::vec3::TreeVec3;

/// Creates a 2-dimensional vector.
#[inline(always)]
#[must_use]
pub fn treevec2(x: Tree, y: Tree) -> TreeVec2 {
    TreeVec2::new(x, y)
}

/// A 2-dimensional vector.
#[derive(Clone, PartialEq)]
pub struct TreeVec2 {
    pub x: Tree,
    pub y: Tree,
}

impl TreeVec2 {
    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub fn new(x: impl Into<Tree>, y: impl Into<Tree>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    #[must_use]
    pub fn splat(v: impl Into<Tree> + Clone) -> Self {
        Self {
            x: v.clone().into(),

            y: v.clone().into(),
        }
    }

    /// Creates a new vector from an array.
    #[inline]
    #[must_use]
    pub fn from_array(a: [impl Into<Tree> + Clone; 2]) -> Self {
        Self::new(a[0].clone().into(), a[1].clone().into())
    }

    /// `[x, y]`
    #[inline]
    #[must_use]
    pub fn to_array(&self) -> [Tree; 2] {
        [self.x.clone(), self.y.clone()]
    }

    /// Creates a vector from the first 2 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 2 elements long.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[impl Into<Tree> + Clone]) -> Self {
        Self::new(slice[0].clone().into(), slice[1].clone().into())
    }

    /// Writes the elements of `self` to the first 2 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 2 elements long.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [Tree]) {
        slice[0] = self.x;
        slice[1] = self.y;
    }

    /// Creates a 3D vector from `self` and the given `z` value.
    #[inline]
    #[must_use]
    pub fn extend(self, z: impl Into<Tree>) -> TreeVec3 {
        TreeVec3::new(self.x, self.y, z.into())
    }

    /// Creates a 2D vector from `self` with the given value of `x`.
    #[inline]
    #[must_use]
    pub fn with_x(mut self, x: impl Into<Tree>) -> Self {
        self.x = x.into();
        self
    }

    /// Creates a 2D vector from `self` with the given value of `y`.
    #[inline]
    #[must_use]
    pub fn with_y(mut self, y: impl Into<Tree>) -> Self {
        self.y = y.into();
        self
    }

    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> Tree {
        (self.x * rhs.x) + (self.y * rhs.y)
    }

    /// Returns a vector where every component is the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self::splat(self.dot(rhs))
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
        self.x.min(self.y)
    }

    /// Returns the horizontal maximum of `self`.
    ///
    /// In other words this computes `max(x, y, ..)`.
    #[inline]
    #[must_use]
    pub fn max_element(self) -> Tree {
        self.x.max(self.y)
    }

    /// Returns the sum of all elements of `self`.
    ///
    /// In other words, this computes `self.x + self.y + ..`.
    #[inline]
    #[must_use]
    pub fn element_sum(self) -> Tree {
        self.x + self.y
    }

    /// Returns the product of all elements of `self`.
    ///
    /// In other words, this computes `self.x * self.y * ..`.
    #[inline]
    #[must_use]
    pub fn element_product(self) -> Tree {
        self.x * self.y
    }

    /// Returns a vector containing the absolute value of each element of `self`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /*
    /// Returns a vector with elements representing the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`
    #[inline]
    #[must_use]
    pub fn signum(self) -> Self {
        Self {

                x: self.x.signum(),
                y: self.y.signum(),
        }
    }
    */

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

    /// Compute the squared euclidean distance between two points in space.
    #[inline]
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> Tree {
        (self - rhs).length_squared()
    }

    /*
    /// Returns the element-wise quotient of [Euclidean division] of `self` by `rhs`.
    #[inline]
    #[must_use]
    pub fn div_euclid(self, rhs: Self) -> Self {
        Self::new(

                self.x.div_euclid(rhs.x),
                self.y.div_euclid(rhs.y),
        )
    }
    */

    /*
    /// Returns the element-wise remainder of [Euclidean division] of `self` by `rhs`.
    ///
    /// [Euclidean division]: Tree::rem_euclid
    #[inline]
    #[must_use]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        Self::new(

                self.x.rem_euclid(rhs.x),
                self.y.rem_euclid(rhs.y),
        )
    }
    */

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
        Self::new(self.x.exp(), self.y.exp())
    }

    /*
    /// Returns a vector containing each element of `self` raised to the power of `n`.
    #[inline]
    #[must_use]
    pub fn powf(self, n: Tree) -> Self {
        Self::new(

                math::powf(self.x, n),
                math::powf(self.y, n),
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
    pub fn lerp(self, rhs: Self, s: impl Into<Tree>) -> Self {
        self.clone() + ((rhs - self) * s.into())
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

    /// Returns a vector that is equal to `self` rotated by 90 degrees.
    #[inline]
    #[must_use]
    pub fn perp(self) -> Self {
        Self {
            x: self.y.neg(),
            y: self.x,
        }
    }

    /// The perpendicular dot product of `self` and `rhs`.
    /// Also known as the wedge product, 2D cross product, and determinant.
    #[doc(alias = "wedge")]
    #[doc(alias = "cross")]
    #[doc(alias = "determinant")]
    #[inline]
    #[must_use]
    pub fn perp_dot(self, rhs: Self) -> Tree {
        (self.x * rhs.y) - (self.y * rhs.x)
    }

    /// Returns `rhs` rotated by the angle of `self`. If `self` is normalized,
    /// then this just rotation. This is what you usually want. Otherwise,
    /// it will be like a rotation with a multiplication by `self`'s length.
    #[inline]
    #[must_use]
    pub fn rotate(self, rhs: Self) -> Self {
        Self {
            x: self.x.clone() * rhs.x.clone() - self.y.clone() * rhs.y.clone(),
            y: self.y.clone() * rhs.x.clone() + self.x.clone() * rhs.y.clone(),
        }
    }
}

impl Default for TreeVec2 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(Tree::constant(0.0))
    }
}

impl Div<TreeVec2> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x.div(rhs.x),
            y: self.y.div(rhs.y),
        }
    }
}

impl DivAssign<TreeVec2> for TreeVec2 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x.div_assign(rhs.x);
        self.y.div_assign(rhs.y);
    }
}

impl Div<Tree> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Tree) -> Self {
        Self {
            x: self.x.div(rhs.clone()),
            y: self.y.div(rhs.clone()),
        }
    }
}

impl DivAssign<Tree> for TreeVec2 {
    #[inline]
    fn div_assign(&mut self, rhs: Tree) {
        self.x.div_assign(rhs.clone());
        self.y.div_assign(rhs.clone());
    }
}

impl Div<TreeVec2> for Tree {
    type Output = TreeVec2;
    #[inline]
    fn div(self, rhs: TreeVec2) -> TreeVec2 {
        TreeVec2 {
            x: self.clone().div(rhs.x),
            y: self.clone().div(rhs.y),
        }
    }
}

impl Mul<TreeVec2> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x.mul(rhs.x),
            y: self.y.mul(rhs.y),
        }
    }
}

impl MulAssign<TreeVec2> for TreeVec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x.mul_assign(rhs.x);
        self.y.mul_assign(rhs.y);
    }
}

impl Mul<Tree> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Tree) -> Self {
        Self {
            x: self.x.mul(rhs.clone()),
            y: self.y.mul(rhs.clone()),
        }
    }
}

impl MulAssign<Tree> for TreeVec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Tree) {
        self.x.mul_assign(rhs.clone());
        self.y.mul_assign(rhs.clone());
    }
}

impl Mul<TreeVec2> for Tree {
    type Output = TreeVec2;
    #[inline]
    fn mul(self, rhs: TreeVec2) -> TreeVec2 {
        TreeVec2 {
            x: self.clone().mul(rhs.x),
            y: self.clone().mul(rhs.y),
        }
    }
}

impl Add<TreeVec2> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x.add(rhs.x),
            y: self.y.add(rhs.y),
        }
    }
}

impl AddAssign<TreeVec2> for TreeVec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x.add_assign(rhs.x);
        self.y.add_assign(rhs.y);
    }
}

impl Add<Tree> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Tree) -> Self {
        Self {
            x: self.x.add(rhs.clone()),
            y: self.y.add(rhs.clone()),
        }
    }
}

impl AddAssign<Tree> for TreeVec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Tree) {
        self.x.add_assign(rhs.clone());
        self.y.add_assign(rhs.clone());
    }
}

impl Add<TreeVec2> for Tree {
    type Output = TreeVec2;
    #[inline]
    fn add(self, rhs: TreeVec2) -> TreeVec2 {
        TreeVec2 {
            x: self.clone().add(rhs.x),
            y: self.clone().add(rhs.y),
        }
    }
}

impl Sub<TreeVec2> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x.sub(rhs.x),
            y: self.y.sub(rhs.y),
        }
    }
}

impl SubAssign<TreeVec2> for TreeVec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: TreeVec2) {
        self.x.sub_assign(rhs.x);
        self.y.sub_assign(rhs.y);
    }
}

impl Sub<Tree> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Tree) -> Self {
        Self {
            x: self.x.sub(rhs.clone()),
            y: self.y.sub(rhs.clone()),
        }
    }
}

impl SubAssign<Tree> for TreeVec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Tree) {
        self.x.sub_assign(rhs.clone());
        self.y.sub_assign(rhs.clone());
    }
}

impl Sub<TreeVec2> for Tree {
    type Output = TreeVec2;
    #[inline]
    fn sub(self, rhs: TreeVec2) -> TreeVec2 {
        TreeVec2 {
            x: self.clone().sub(rhs.x),
            y: self.clone().sub(rhs.y),
        }
    }
}

impl Rem<TreeVec2> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self {
            x: self.x.modulo(rhs.x),
            y: self.y.modulo(rhs.y),
        }
    }
}

impl RemAssign<TreeVec2> for TreeVec2 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.x = self.x.modulo(rhs.x);
        self.y = self.y.modulo(rhs.y);
    }
}

impl Rem<Tree> for TreeVec2 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Tree) -> Self {
        Self {
            x: self.x.modulo(rhs.clone()),
            y: self.y.modulo(rhs.clone()),
        }
    }
}

impl RemAssign<Tree> for TreeVec2 {
    #[inline]
    fn rem_assign(&mut self, rhs: Tree) {
        self.x = self.x.modulo(rhs.clone());
        self.y = self.y.modulo(rhs.clone());
    }
}

impl Rem<TreeVec2> for Tree {
    type Output = TreeVec2;
    #[inline]
    fn rem(self, rhs: TreeVec2) -> TreeVec2 {
        TreeVec2 {
            x: self.clone().modulo(rhs.x),
            y: self.clone().modulo(rhs.y),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[Tree; 2]> for TreeVec2 {
    #[inline]
    fn as_ref(&self) -> &[Tree; 2] {
        unsafe { &*(self as *const TreeVec2 as *const [Tree; 2]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[Tree; 2]> for TreeVec2 {
    #[inline]
    fn as_mut(&mut self) -> &mut [Tree; 2] {
        unsafe { &mut *(self as *mut TreeVec2 as *mut [Tree; 2]) }
    }
}

impl Sum for TreeVec2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::splat(Tree::constant(0.0)), Self::add)
    }
}

impl<'a> Sum<&'a Self> for TreeVec2 {
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

impl Product for TreeVec2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::splat(Tree::constant(1.0)), Self::mul)
    }
}

impl<'a> Product<&'a Self> for TreeVec2 {
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

impl Neg for TreeVec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: self.x.neg(),
            y: self.y.neg(),
        }
    }
}

impl Index<usize> for TreeVec2 {
    type Output = Tree;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for TreeVec2 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for TreeVec2 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple(stringify!(TreeVec2))
            .field(&self.x)
            .field(&self.y)
            .finish()
    }
}

impl From<[Tree; 2]> for TreeVec2 {
    #[inline]
    fn from(a: [Tree; 2]) -> Self {
        Self::new(a[0].clone(), a[1].clone())
    }
}

impl From<TreeVec2> for [Tree; 2] {
    #[inline]
    fn from(v: TreeVec2) -> Self {
        [v.x, v.y]
    }
}

impl From<(Tree, Tree)> for TreeVec2 {
    #[inline]
    fn from(t: (Tree, Tree)) -> Self {
        Self::new(t.0, t.1)
    }
}

impl From<TreeVec2> for (Tree, Tree) {
    #[inline]
    fn from(v: TreeVec2) -> Self {
        (v.x, v.y)
    }
}
