use std::f64::consts::TAU;
use num_traits::identities::Zero;

pub type num = f64;
pub use num_complex::Complex64 as scalar;

pub mod cairo_util;
pub mod clock;
pub mod tixy;
