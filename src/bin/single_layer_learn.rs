extern crate card_engine;
extern crate rand;
extern crate ndarray;
#[macro_use]
extern crate lazy_static;

use ndarray::prelude::*;
use card_engine::learning::neural_net::*;
use rand::{Rng, thread_rng};

const FL: ActivationFunction = ActivationFunction::SymmetricSigmoid;

fn f(v: &ArrayView<f32, Ix1>) -> f32 {
    lazy_static! {
        static ref A: Array<f32, Ix1> = arr1(&[0.3, 0.4, -0.5, 0.8, 0.2]);
    }
    FL.af()(v.dot(&A) - 0.11)
}

fn main() {
    let layers = [LayerDesc::new(5, 1, FL)];
    let mut nn = NeuralNet::new(&layers, 0.1).unwrap();

    let mut rng = thread_rng();
    let mut grad = Array::zeros(nn.num_parameters());

    for i in 0..1000 {
        let sample: Vec<_> = (0..5).map(|_| { rng.gen_range(-1.0, 1.0) }).collect();
        let v = aview1(&sample);
        let out = f(&v);

        let nn_out = nn.evaluate_with_gradient(&v, grad.view_mut())[0];

        let err = out - nn_out;
        if i % 99 == 0 {
            println!("Iteration {} Err = {}", i+1, err);
        }
        nn.update_weights(err, grad.view());
    }
}
