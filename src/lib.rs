extern crate rand;
extern crate termion;
#[macro_use]
extern crate lazy_static;
extern crate atty;
extern crate ndarray;
extern crate ndarray_rand;
extern crate itertools;

pub mod germanwhist;
pub mod cards;
pub mod learning;

pub use germanwhist::engine::{Round, Action, ActionError};
pub use germanwhist::state::{GameState};
pub use learning::neural_net::{NeuralNet, LayerDesc, OutputFunction};
use cards::{Suit, Rank};
