// Generated from vec.rs.tera template. Edit the template, not the generated file.

#[cfg(not(target_arch = "spirv"))]
use core::fmt;

use core::ops::*;
use fidget::context::Tree;

/// Wrapper around the Tree type which reduces the amount of cloning needed.
#[derive(Clone, PartialEq)]
pub struct TreeFloat {
    pub x: Tree,
}

impl TreeFloat {
    /// Creates a new vector.
    #[inline(always)]
    #[must_use]
    pub fn new(x: impl Into<Tree>) -> Self {
        Self { x: x.into() }
    }

    /// Creates a new vector from an array.
    #[inline]
    #[must_use]
    pub fn from_array(a: [impl Into<Tree> + Clone; 1]) -> Self {
        Self::new(a[0].clone().into())
    }

    /// `[x]`
    #[inline]
    #[must_use]
    pub fn to_array(&self) -> [Tree; 1] {
        [self.x.clone()]
    }

    /// Creates a vector from the first 1 values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 1 elements long.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[impl Into<Tree> + Clone]) -> Self {
        Self::new(slice[0].clone().into())
    }

    /// Writes the elements of `self` to the first 1 elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than 1 elements long.
    #[inline]
    pub fn write_to_slice(&self, slice: &mut [Tree]) {
        slice[0] = self.x.clone();
    }

    /// Creates a 1D vector from `self` with the given value of `x`.
    #[inline]
    #[must_use]
    pub fn with_x(mut self, x: impl Into<Tree>) -> Self {
        self.x = x.into();
        self
    }

    /// Computes the dot product of `self` and `rhs`.
    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> Tree {
        self.x.clone() * rhs.x.clone()
    }

    /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[self.x.min(rhs.x), self.y.min(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn min(&self, rhs: &Self) -> Self {
        Self {
            x: self.x.min(rhs.x.clone()),
        }
    }

    /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
    ///
    /// In other words this computes `[self.x.max(rhs.x), self.y.max(rhs.y), ..]`.
    #[inline]
    #[must_use]
    pub fn max(&self, rhs: &Self) -> Self {
        Self {
            x: self.x.max(rhs.x.clone()),
        }
    }

    /// Component-wise clamping of values, similar to [`Tree::clamp`].
    ///
    /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
    #[inline]
    #[must_use]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns a vector containing the absolute value of each element of `self`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.clone().abs(),
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
    pub fn signum(&self) -> Self {
        Self {

                x: self.x.clone().signum(),
        }
    }
    */

    /// Computes the length of `self`.
    #[doc(alias = "magnitude")]
    #[inline]
    #[must_use]
    pub fn length(&self) -> Tree {
        self.clone().dot(self).sqrt()
    }

    /// Computes the squared length of `self`.
    ///
    /// This is faster than `length()` as it avoids a square root operation.
    #[doc(alias = "magnitude2")]
    #[inline]
    #[must_use]
    pub fn length_squared(&self) -> Tree {
        self.clone().dot(self)
    }

    /// Computes `1.0 / length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    #[inline]
    #[must_use]
    pub fn length_recip(&self) -> Tree {
        Tree::constant(1.0) / self.length()
    }

    /// Computes the Euclidean distance between two points in space.
    #[inline]
    #[must_use]
    pub fn distance(&self, rhs: &Self) -> Tree {
        (self.clone() - rhs).length()
    }

    /// Compute the squared euclidean distance between two points in space.
    #[inline]
    #[must_use]
    pub fn distance_squared(&self, rhs: &Self) -> Tree {
        (self.clone() - rhs).length_squared()
    }

    /*
    /// Returns the element-wise quotient of [Euclidean division] of `self` by `rhs`.
    #[inline]
    #[must_use]
    pub fn div_euclid(&self, rhs: &Self) -> Self {
        Self::new(

                self.x.div_euclid(rhs.x),
        )
    }
    */

    /*
    /// Returns the element-wise remainder of [Euclidean division] of `self` by `rhs`.
    ///
    /// [Euclidean division]: Tree::rem_euclid
    #[inline]
    #[must_use]
    pub fn rem_euclid(&self, rhs: &Self) -> Self {
        Self::new(

                self.x.rem_euclid(rhs.x),
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
    pub fn normalize(&self) -> Self {
        #[allow(clippy::let_and_return)]
        let normalized = self.clone().mul(&self.length_recip());
        normalized
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be of non-zero length.
    #[inline]
    #[must_use]
    pub fn project_onto(&self, rhs: &Self) -> Self {
        let other_len_sq_rcp = Tree::constant(1.0) / rhs.clone().dot(rhs);
        rhs.clone() * &self.dot(rhs) * &other_len_sq_rcp
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be of non-zero length.
    #[inline]
    #[must_use]
    pub fn reject_from(&self, rhs: &Self) -> Self {
        self.clone() - &self.project_onto(rhs)
    }

    /// Returns the vector projection of `self` onto `rhs`.
    ///
    /// `rhs` must be normalized.
    #[inline]
    #[must_use]
    pub fn project_onto_normalized(&self, rhs: &Self) -> Self {
        rhs.clone() * &self.dot(rhs)
    }

    /// Returns the vector rejection of `self` from `rhs`.
    ///
    /// The vector rejection is the vector perpendicular to the projection of `self` onto
    /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
    ///
    /// `rhs` must be normalized.
    #[inline]
    #[must_use]
    pub fn reject_from_normalized(&self, rhs: &Self) -> Self {
        self.clone() - &self.project_onto_normalized(rhs)
    }

    /// Returns a vector containing the nearest integer to a number for each element of `self`.
    /// Round half-way cases away from 0.0.
    #[inline]
    #[must_use]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.clone().round(),
        }
    }

    /// Returns a vector containing the largest integer less than or equal to a number for each
    /// element of `self`.
    #[inline]
    #[must_use]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.clone().floor(),
        }
    }

    /// Returns a vector containing the smallest integer greater than or equal to a number for
    /// each element of `self`.
    #[inline]
    #[must_use]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.clone().ceil(),
        }
    }

    /*
    /// Returns a vector containing the integer part each element of `self`. This means numbers are
    /// always truncated towards zero.
    #[inline]
    #[must_use]
    pub fn trunc(&self) -> Self {
        Self {

                x: self.x.clone().trunc(),
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
    pub fn fract(&self) -> Self {
        self.clone() - self.trunc()
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
    pub fn fract_gl(&self) -> Self {
        self.clone() - &self.floor()
    }

    /// Returns a vector containing `e^self` (the exponential function) for each element of
    /// `self`.
    #[inline]
    #[must_use]
    pub fn exp(&self) -> Self {
        Self::new(self.x.clone().exp())
    }

    /*
    /// Returns a vector containing each element of `self` raised to the power of `n`.
    #[inline]
    #[must_use]
    pub fn powf(&self, n: Tree) -> Self {
        Self::new(

                math::powf(self.x, n),
        )
    }
    */

    /// Returns a vector containing the reciprocal `1.0/n` of each element of `self`.
    #[inline]
    #[must_use]
    pub fn recip(&self) -> Self {
        Self {
            x: Tree::constant(1.0) / self.x.clone(),
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
    pub fn lerp(&self, rhs: &Self, s: &Tree) -> Self {
        self.clone() + &((rhs.clone() - self) * s)
    }

    /// Calculates the midpoint between `self` and `rhs`.
    ///
    /// The midpoint is the average of, or halfway point between, two vectors.
    /// `a.midpoint(b)` should yield the same result as `a.lerp(b, 0.5)`
    /// while being slightly cheaper to compute.
    #[inline]
    pub fn midpoint(&self, rhs: &Self) -> Self {
        (self.clone() + rhs) * &Tree::constant(0.5)
    }
}

impl Default for TreeFloat {
    #[inline(always)]
    fn default() -> Self {
        Self::new(Tree::constant(0.0))
    }
}

impl Div<&TreeFloat> for TreeFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &Self) -> Self {
        Self {
            x: self.x.div(rhs.x.clone()),
        }
    }
}

impl DivAssign<&TreeFloat> for TreeFloat {
    #[inline]
    fn div_assign(&mut self, rhs: &Self) {
        self.x.div_assign(rhs.x.clone());
    }
}

impl Div<&Tree> for TreeFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &Tree) -> Self {
        Self {
            x: self.x.div(rhs.clone()),
        }
    }
}

impl DivAssign<&Tree> for TreeFloat {
    #[inline]
    fn div_assign(&mut self, rhs: &Tree) {
        self.x.div_assign(rhs.clone());
    }
}

impl Mul<&TreeFloat> for TreeFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        Self {
            x: self.x.mul(rhs.x.clone()),
        }
    }
}

impl MulAssign<&TreeFloat> for TreeFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        self.x.mul_assign(rhs.x.clone());
    }
}

impl Mul<&Tree> for TreeFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Tree) -> Self {
        Self {
            x: self.x.mul(rhs.clone()),
        }
    }
}

impl MulAssign<&Tree> for TreeFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: &Tree) {
        self.x.mul_assign(rhs.clone());
    }
}

impl Add<&TreeFloat> for TreeFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        Self {
            x: self.x.add(rhs.x.clone()),
        }
    }
}

impl AddAssign<&TreeFloat> for TreeFloat {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.x.add_assign(rhs.x.clone());
    }
}

impl Add<&Tree> for TreeFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Tree) -> Self {
        Self {
            x: self.x.add(rhs.clone()),
        }
    }
}

impl AddAssign<&Tree> for TreeFloat {
    #[inline]
    fn add_assign(&mut self, rhs: &Tree) {
        self.x.add_assign(rhs.clone());
    }
}

impl Sub<&TreeFloat> for TreeFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        Self {
            x: self.x.sub(rhs.x.clone()),
        }
    }
}

impl SubAssign<&TreeFloat> for TreeFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: &TreeFloat) {
        self.x.sub_assign(rhs.x.clone());
    }
}

impl Sub<&Tree> for TreeFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Tree) -> Self {
        Self {
            x: self.x.sub(rhs.clone()),
        }
    }
}

impl SubAssign<&Tree> for TreeFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: &Tree) {
        self.x.sub_assign(rhs.clone());
    }
}

impl Rem<&TreeFloat> for TreeFloat {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: &Self) -> Self {
        Self {
            x: self.x.modulo(rhs.x.clone()),
        }
    }
}

impl RemAssign<&TreeFloat> for TreeFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: &Self) {
        self.x = self.x.modulo(rhs.x.clone());
    }
}

impl Rem<&Tree> for TreeFloat {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: &Tree) -> Self {
        Self {
            x: self.x.modulo(rhs.clone()),
        }
    }
}

impl RemAssign<&Tree> for TreeFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: &Tree) {
        self.x = self.x.modulo(rhs.clone());
    }
}

impl Rem<&TreeFloat> for &Tree {
    type Output = TreeFloat;
    #[inline]
    fn rem(self, rhs: &TreeFloat) -> TreeFloat {
        TreeFloat {
            x: self.clone().modulo(rhs.x.clone()),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsRef<[Tree; 1]> for TreeFloat {
    #[inline]
    fn as_ref(&self) -> &[Tree; 1] {
        unsafe { &*(self as *const TreeFloat as *const [Tree; 1]) }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl AsMut<[Tree; 1]> for TreeFloat {
    #[inline]
    fn as_mut(&mut self) -> &mut [Tree; 1] {
        unsafe { &mut *(self as *mut TreeFloat as *mut [Tree; 1]) }
    }
}

impl Neg for TreeFloat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: self.x.neg() }
    }
}

impl Index<usize> for TreeFloat {
    type Output = Tree;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for TreeFloat {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            _ => panic!("index out of bounds"),
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Debug for TreeFloat {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple(stringify!(TreeFloat))
            .field(&self.x)
            .finish()
    }
}

impl From<[Tree; 1]> for TreeFloat {
    #[inline]
    fn from(a: [Tree; 1]) -> Self {
        Self::new(a[0].clone())
    }
}

impl From<TreeFloat> for [Tree; 1] {
    #[inline]
    fn from(v: TreeFloat) -> Self {
        [v.x]
    }
}

impl From<(Tree,)> for TreeFloat {
    #[inline]
    fn from(t: (Tree,)) -> Self {
        Self::new(t.0)
    }
}

impl From<TreeFloat> for (Tree,) {
    #[inline]
    fn from(v: TreeFloat) -> Self {
        (v.x,)
    }
}

impl From<Tree> for TreeFloat {
    #[inline]
    fn from(t: Tree) -> Self {
        Self::new(t)
    }
}

impl From<TreeFloat> for Tree {
    #[inline]
    fn from(v: TreeFloat) -> Self {
        v.x
    }
}
