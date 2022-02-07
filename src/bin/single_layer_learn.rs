#![allow(unused)]

use once_cell::sync::Lazy;
use ndarray::prelude::*;
use card_engine::learning::neural_net::*;
use rand::{Rng, thread_rng};
use ndarray::linalg::general_mat_vec_mul;
use std::cmp;
use clap::{App, Arg};

const FL: ActivationFunction = ActivationFunction::Sigmoid;

fn f1(v: &ArrayView<f32, Ix1>) -> Array<f32, Ix1> {
    static A: Lazy<Array<f32, Ix1>> = Lazy::new(|| arr1(&[0.3, 0.4, -0.5, 0.8, 0.2]));
    arr1(&[FL.af()(v.dot(&A) - 0.11)])
}

fn f1w(v: &ArrayView<f32, Ix1>) -> Array<f32,Ix1> {
    static A: Lazy<Array<f32, Ix2>> = Lazy::new(|| arr2(&[[0.3, 0.4, -0.5, 0.8, 0.2],
                                               [-0.2, 0.6, 0.3, 0.1, 0.1]]));
    static B: Lazy<Array<f32, Ix1>> = Lazy::new(|| arr1(&[-0.071, 0.4]));;
    let f = FL.af();
    let mut x = B.to_owned();
    general_mat_vec_mul(1.0, &A, &v, 1.0, &mut x);
    x.map(|r| f(*r))
}


fn f2(v: &ArrayView<f32, Ix1>) -> Array<f32, Ix1> {
    static A1: Lazy<Array<f32, Ix2>> = Lazy::new(|| arr2(&[[0.3, 0.4, -0.5, 0.8, 0.2],
                                                [0.14, -0.4, -0.5, 0.28, 0.2],
                                                [0.56, 0.4, 0.75, -0.1, -0.2]]));
    static B1: Lazy<Array<f32, Ix1>> = Lazy::new(|| arr1(&[0.3, 0.4, -0.5]));

    static A2: Lazy<Array<f32, Ix1>> = Lazy::new(|| arr1(&[1.0, -0.2, 0.5]));

    const B2: f32 = -0.12;

    let mut l1: ArrayBase<_, Ix1> = B1.to_owned();
    general_mat_vec_mul(1.0, &A1, &v, 1.0, &mut l1);
    let x1 = l1.map(|x| FL.af()(*x));
    arr1(&[FL.af()(x1.dot(&A2) + B2)])
}

fn f2s(v: &ArrayView<f32, Ix1>) -> Array<f32, Ix1> {
    static A1: Lazy<Array<f32, Ix2>> = Lazy::new(|| arr2(&[[0.1, 0.2],
                                                [0.3, 0.4],
                                                [0.5, 0.6]]));
    static B1: Lazy<Array<f32, Ix1>> = Lazy::new(|| arr1(&[0.7, 0.8, 0.9]));;

    static A2: Lazy<Array<f32, Ix1>> = Lazy::new(|| arr1(&[0.6, -0.4, -0.2]));;
    const B2: f32 = -0.12;

    let mut l1: ArrayBase<_, Ix1> = B1.to_owned();
    general_mat_vec_mul(1.0, &A1, &v, 1.0, &mut l1);
    let x1 = l1.map(|x| FL.af()(*x));
    arr1(&[FL.af()(x1.dot(&A2) + B2)])
}

#[allow(unused)]
fn train_single_layer(iter: usize) {
    let layers = [LayerDesc::new(5, 1, FL)];
    let nn = NeuralNet::new(&layers, 0.1).unwrap();

    train_nn(nn, f1, iter);
}

fn train_single_wide_layer(iter: usize) {
    let layers = [LayerDesc::new(5, 2, FL)];
    let nn = NeuralNet::new(&layers, 0.01).unwrap();

    train_nn(nn, f1w, iter);
}

fn train_dual_layer(iter: usize) {
    let layers = [LayerDesc::new(5, 3, FL), LayerDesc::new(3, 1, FL)];
    let nn = NeuralNet::new(&layers, 0.01).unwrap();

    //debug_nn(nn, f2);
    train_nn(nn, f2, iter);
}

fn train_dual_small_layer(iter: usize) {
    let layers = [LayerDesc::new(2, 3, FL), LayerDesc::new(3, 1, FL)];
    let nn = NeuralNet::new(&layers, 0.1).unwrap();

    //debug_nn(nn, f2s);
    train_nn(nn, f2s, iter);
}


fn debug_nn(mut nn: NeuralNet, target: fn(&ArrayView<f32, Ix1>) -> Array<f32, Ix1>) {
    let mut rng = thread_rng();
    let mut grad = Array::zeros(nn.num_parameters());

    //nn.dump();

    let sample: Vec<_> = (0..nn.num_inputs())
        .map(|_| rng.gen_range(-1.0, 1.0))
        .collect();
    let v = aview1(&sample);
    let out = target(&v);

    let nn_out = nn.evaluate_with_gradient(&v, grad.view_mut())[0];

    let (nn_1, nn_2) = nn.clone().split_at(1);
    let mut grad_1 = Array::zeros(nn_1.num_parameters());
    let nn_out_1 = nn_1.evaluate_with_gradient(&v, grad_1.view_mut());

    let mut grad_2 = Array::zeros(nn_2.num_parameters());
    let nn_out_2 = nn_2.evaluate_with_gradient(&nn_out_1, grad_2.view_mut());

    println!("Samp: {:8.5}", v);
    println!("SaL1: {:8.5}", nn_out_1);
    println!("SaL2: {:8.5}", nn_out_2);
    println!("Weig: {:8.5}", nn.weights());
    println!("Grad: {:8.5}", grad);
}


fn train_nn(mut nn: NeuralNet, target: fn(&ArrayView<f32, Ix1>) -> Array<f32, Ix1>, niter: usize) {
    let mut rng = thread_rng();
    let mut grad = Array::zeros(nn.num_parameters());


    let op = cmp::max(niter / 20, 1);
    for i in 0..niter {
        let sample: Vec<_> = (0..nn.num_inputs())
            .map(|_| rng.gen_range(-1.0, 1.0))
            .collect();

        let v = aview1(&sample);
        let out = target(&v);

        let nn_out = nn.evaluate_with_gradient(&v, grad.view_mut());
        let nn_out_nog = nn.evaluate(&v);

        let err_v = out.to_owned() - &nn_out;
        let err = (err_v.dot(&err_v)).sqrt();

        if i % op == op - 1 || i == 0 {
            println!("Weights: {:8.5}", nn.weights());
            println!("Iteration {:8}, Target {:8.5}, Predict {:8.5} / {:8.5}, |Err| = {:8.5}, |G| = {:8.5}",
                     i + 1,
                     out,
                     nn_out,
                     nn_out_nog,
                     err, grad.dot(&grad).sqrt() * err);
        }
        nn.update_weights(err_v[0], grad.view());
    }
    println!("Weights: {:8.5}", nn.weights());
}

fn main() {
    let m = App::new("x")
        .arg(Arg::with_name("ITER").required(true).index(1))
        .arg(Arg::with_name("FUNC").required(true).index(2)).get_matches();

    let fi = m.value_of("FUNC").map(|x| x.parse::<usize>().ok().unwrap()).unwrap_or(1);
    let f = match fi {
        1 => train_single_layer,
        2 => train_single_wide_layer,
        3 => train_dual_layer,
        4 => train_dual_small_layer,
        _ => panic!("bad entry")
    };

    f(m.value_of("ITER").map(|x| x.parse::<usize>().ok().unwrap()).unwrap_or(10000));
}
