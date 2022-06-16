use std::io::Write;

/// Create a Cairo image surface with the size provided. The program
/// will crash if there is a problem.
pub fn create_image_surface(
  width: i32,
  height: i32,
) -> cairo::ImageSurface {
  let surface = cairo::ImageSurface::create(
    cairo::Format::Rgb24,
    width,
    height,
  ).expect("Couldn't create image surface!");
  return surface;
}

/// Get the Cairo drawing context associated with the image
/// provided. The program will crash if there is a problem.
pub fn get_image_context(
  image: &cairo::ImageSurface,
) -> cairo::Context {
  let context = cairo::Context::new(&image)
    .expect("Couldn't create image context!");
  return context;
}

/// Write the BGRA pixel data of an image surface to the stream
/// provided. The program will crash if there is a problem.
pub fn write_image_surface<W: Write>(
  image: &mut cairo::ImageSurface,
  stream: &mut W,
) {
  let data = image.data()
    .expect("Couldn't read image data!");
  stream.write_all(&*data)
    .expect("Couldn't write image data!");
}
