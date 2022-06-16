use std::f64::consts::TAU;
use num_traits::identities::Zero;

use crate::num;
use crate::scalar;
use crate::clock::Clock;
use crate::cairo_util::create_image_surface;
use crate::cairo_util::write_image_surface;
use crate::cairo_util::get_image_context;

/// Draw a Tixy-style color field with the parameters given. This
/// function assumes the context is in normalized device coordinates.
fn tixy(
  // The scalar-to-scalar function that defines the color field.
  model: &dyn Fscalar,
  // The current time, in seconds.
  time: num,
  // The number of rows to draw.
  rows: usize,
  // The number of columns to draw.
  cols: usize,
  // The maximum size of a dot drawn by the shader.
  max_radius: num,
  // The Cairo drawing context.
  ctx: &cairo::Context,
) {
  // Draw a border.
  set_color_white(&ctx);
  ctx.new_path();
  ctx.set_line_width(2.0/256.0);
  ctx.rectangle(-1.0, -1.0, 2.0, 2.0);
  ctx.stroke().unwrap();
  // Put the contents of the shader a little bit inside the border.
  ctx.save().unwrap(); {
    ctx.scale(0.95, 0.95);
    for row in 0..rows {
      let dy = row as num / rows as num;
      for col in 0..cols {
        let dx = col as num / cols as num;
        // Transform from [0..col, 0..row] space to [-1..1, -1..1]
        // normalized device coordinates.
        let ndc_x = 2.0*dx-1.0+max_radius;
        let ndc_y = 2.0*dy-1.0+max_radius;
        let source = scalar::new(ndc_x, ndc_y);
        let source = scalar::from_polar(
          source.norm(),
          source.arg()+time*TAU,
        );
        let mut target = scalar::zero();
        // The fractal flame algorithm makes me think it might be
        // interesting to make Tixy-style shaders iterative. This
        // would be more meaningful in the context of Markov
        // algorithms and the new language I want to work on.
        let iterations = 100;
        for _ in 0..iterations {
          target = model.eval(source);
        }
        // The standard Tixy app draws white if the value is below
        // zero and red if the value is above zero. Since I use
        // complex numbers, I want the color to depend on the value's
        // phase instead.
        if target.arg() < 0.0 {
          set_color_white(&ctx);
        } else {
          set_color_super_pink(&ctx);
        }
        // Tweak the radius given by the model.
        //let radius = target.norm()*1.0*0.125;
        //let radius = radius.min(max_radius);
        let range = 2.0;
        let length = target.norm().min(range)/range;
        let radius = length*max_radius;
        // For now, just draw a circle.
        ctx.new_path();
        ctx.arc(ndc_x, ndc_y, radius, 0.0, TAU);
        ctx.fill().unwrap();
      }
    }
  } ctx.restore().unwrap();
}

// TODO: think of a better api for drawing
pub fn draw_sketch_2022_06_16() {
  let width = 1024;
  let height = 1024;
  let ww = width as num * 0.5;
  let hh = height as num * 0.5;
  let length = 6.0;
  let framerate = 15;
  let rows = 16;
  let cols = 16;
  let max_radius = 2.0 / rows as num / 2.0;
  let model = ResNet::with_depth(2);
  let mut clock = Clock::new(length, framerate);
  let mut img = create_image_surface(width, height);
  eprintln!(
    "tixy: {}x{}px {}fps {}sec",
    width,
    height,
    framerate,
    length,
  );
  while clock.is_some() {
    {
      let time = clock.t() / length as num;
      let ctx = get_image_context(&img);
      // Clear the display.
      set_color_space_cadet(&ctx);
      ctx.paint().unwrap();
      ctx.save().unwrap(); {
        // Use normalized device coordinates. The center of the
        // display is the origin of the coordinate system, and the
        // edges are a distance of one away.
        ctx.translate(ww, hh);
        ctx.scale(ww, hh);
        ctx.scale(0.95, 0.95);
        // Draw a border.
        ctx.set_source_rgb(1.0, 1.0, 1.0);
        ctx.new_path();
        ctx.set_line_width(2.0/256.0);
        ctx.rectangle(-1.0, -1.0, 2.0, 2.0);
        ctx.stroke().unwrap();
        /* Draw a tixy-style shader in the upper-left quadrant. */
        ctx.save().unwrap(); {
          ctx.translate(-0.5, -0.5);
          ctx.scale(0.5, 0.5);
          tixy(&model, time, rows, cols, max_radius, &ctx);
        } ctx.restore().unwrap();
        ctx.save().unwrap(); {
          ctx.translate(0.5, -0.5);
          ctx.scale(0.5, 0.5);
          ctx.scale(-1.0, 1.0);
          tixy(&model, time, rows, cols, max_radius, &ctx);
        } ctx.restore().unwrap();
        ctx.save().unwrap(); {
          ctx.translate(-0.5, 0.5);
          ctx.scale(0.5, 0.5);
          ctx.scale(1.0, -1.0);
          tixy(&model, time, rows, cols, max_radius, &ctx);
        } ctx.restore().unwrap();
        ctx.save().unwrap(); {
          ctx.translate(0.5, 0.5);
          ctx.scale(0.5, 0.5);
          ctx.scale(-1.0, -1.0);
          tixy(&model, time, rows, cols, max_radius, &ctx);
        } ctx.restore().unwrap();
      } ctx.restore().unwrap();
    }
    let mut stdout = std::io::stdout();
    write_image_surface(&mut img, &mut stdout);
    clock.tick();
  }
}

fn set_color_super_pink(
  cx: &cairo::Context,
) {
  cx.set_source_rgb(0.839, 0.360, 0.678);
}

fn set_color_space_cadet(
  cx: &cairo::Context,
) {
  cx.set_source_rgb(0.160, 0.160, 0.239);
}

fn set_color_white(
  cx: &cairo::Context,
) {
  cx.set_source_rgb(1.0, 1.0, 1.0);
}

fn set_color_black(
  cx: &cairo::Context,
) {
  cx.set_source_rgb(0.0, 0.0, 0.0);
}

/// Return a random complex number, with a magnitude between 0 and 1,
/// and a phase between 0 and tau.
fn rand_scalar() -> scalar {
  let len = rand::random::<num>().sqrt();
  let phi = rand::random::<num>()*TAU;
  return scalar::from_polar(len, phi);
}

/// A function from one scalar to another.
trait Fscalar {
  fn eval(&self, src: scalar) -> scalar;
}

/// A residual network, used to define random scalar fields.
struct ResNet {
  weight: Vec<scalar>,
}

impl ResNet {
  /// Create a residual network the number of layers provided.
  fn with_depth(depth: usize) -> Self {
    let mut weight = Vec::new();
    for _ in 0..depth {
      let linear = rand_scalar();
      let bias = rand_scalar();
      let threshold = rand_scalar();
      weight.push(linear);
      weight.push(bias);
      weight.push(threshold);
    }
    ResNet { weight }
  }
}

impl Fscalar for ResNet {
  fn eval(&self, mut value: scalar) -> scalar {
    let depth = self.weight.len()/3;
    for i in 0..depth {
      let linear = self.weight[3*i+0];
      let bias = self.weight[3*i+1];
      let threshold = self.weight[3*i+2];
      let mut residual = value*linear+bias;
      if residual.norm_sqr() < threshold.norm_sqr() {
        residual = scalar::zero();
      } else {
        residual = scalar::from_polar(
          residual.norm_sqr(),
          residual.arg()+threshold.arg(),
        );
      }
      value += residual;
    }
    return value;
  }
}
