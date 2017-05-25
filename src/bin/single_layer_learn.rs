extern crate card_engine;
extern crate rand;
extern crate ndarray;
#[macro_use]
extern crate lazy_static;

use ndarray::prelude::*;
use card_engine::learning::neural_net::*;
use rand::{Rng, thread_rng};
use ndarray::linalg::general_mat_vec_mul;

const FL: ActivationFunction = ActivationFunction::Linear;

fn f1(v: &ArrayView<f32, Ix1>) -> f32 {
    lazy_static! {
        static ref A: Array<f32, Ix1> = arr1(&[0.3, 0.4, -0.5, 0.8, 0.2]);
    }
    FL.af()(v.dot(&A) - 0.11)
}

fn f2(v: &ArrayView<f32, Ix1>) -> f32 {
    lazy_static! {
        static ref A1: Array<f32, Ix2> = arr2(&[[0.3, 0.4, -0.5, 0.8, 0.2],
                                                [0.14, -0.4, -0.5, 0.28, 0.2],
                                                [0.56, 0.4, 0.75, -0.1, -0.2]]);
        static ref B1: Array<f32, Ix1> = arr1(&[0.3, 0.4, -0.5]);

        static ref A2: Array<f32, Ix1> = arr1(&[0.3, 0.4, -0.5]);
    }
    const B2: f32 = -0.12;

    let mut l1: ArrayBase<_, Ix1> = B1.to_owned();
    general_mat_vec_mul(1.0, &A1, &v, 1.0, &mut l1);
    let x1 = l1.map(|x| FL.af()(*x));
    FL.af()(x1.dot(&A2) + B2)
}

#[allow(unused)]
fn train_single_layer() {
    let layers = [LayerDesc::new(5, 1, FL)];
    let nn = NeuralNet::new(&layers, 0.1).unwrap();

    train_nn(nn, f1);
}

fn train_dual_layer() {
    let layers = [LayerDesc::new(5, 3, FL), LayerDesc::new(3, 1, FL)];
    let nn = NeuralNet::new(&layers, 0.1).unwrap();

    train_nn(nn, f2);
}

fn train_nn(mut nn: NeuralNet, target: fn(&ArrayView<f32, Ix1>) -> f32) {
    let mut rng = thread_rng();
    let mut grad = Array::zeros(nn.num_parameters());

    nn.dump();

    for i in 0..1000 {
        let sample: Vec<_> = (0..5).map(|_| { rng.gen_range(-1.0, 1.0) }).collect();
        let v = aview1(&sample);
        let out = target(&v);

        let nn_out = nn.evaluate_with_gradient(&v, grad.view_mut())[0];
        let nn_out_nog = nn.evaluate(&v)[0];

        let err = out - nn_out;
        println!("*****\n{}, {}\n*******", err, grad);

        if i % 100 == 99 {
            println!("Iteration {}, Target {}, Predict {} / {}, Err = {}", i+1, out, nn_out, nn_out_nog, err);
        }
        nn.update_weights(err, grad.view());
        println!("{}", nn.weights());
    }
}

fn main() {
    train_dual_layer()
}
