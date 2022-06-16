use crate::num;

/// The clock records the total duration of a piece, the current
/// position in time, and the rate at which computation is done.
pub struct Clock {
  /// The duration of the piece, in seconds.
  length: num,
  /// The current frame of the piece.
  frame: usize,
  /// The duration of the piece, in frames.
  frames: usize,
  /// The sample rate of the piece, in frames per second.
  framerate: usize,
}

impl Clock {
  /// Create a new clock, at time zero, with a duration in seconds and
  /// a sample rate in frames per second.
  pub fn new(
    length: num,
    framerate: usize,
  ) -> Self {
    let dt = 1.0 / framerate as num;
    let frames = length as usize * framerate;
    Clock {
      length,
      frame: 0,
      frames,
      framerate,
    }
  }

  /// Returns true if the clock's current position in time is less
  /// than the clock's total duration.
  pub fn is_some(&self) -> bool {
    return self.frame < self.frames;
  }

  /// Returns true if the clock's current position in time is greater
  /// than or equal to the clock's total duration.
  pub fn is_none(&self) -> bool {
    return self.frame >= self.frames;
  }

  /// The clock's current position in time, in seconds.
  pub fn t(&self) -> num {
    let frame = self.frame as num;
    let framerate = self.framerate as num;
    return frame / framerate;
  }

  /// The clock's sample rate in seconds.
  pub fn dt(&self) -> num {
    let framerate = self.framerate as num;
    return 1.0 / framerate;
  }

  /// The clock's total duration in time.
  pub fn len(&self) -> num {
    return self.length;
  }

  /// Advance the clock's position in time by one frame.
  pub fn tick(&mut self) {
    self.frame += 1;
  }
}
