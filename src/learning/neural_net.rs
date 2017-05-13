use ndarray::prelude::*;
use ndarray::{Data, DataMut, Zip};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use itertools::{Itertools};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    SymmetricSigmoid,
    ReLU
}

fn activate_linear(x: f32) -> f32 { x }
fn activate_logistic(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
const SS_FSCALE: f32 = 1.7159;
const SS_XSCALE: f32 = 2.0/3.0;
fn activate_ss(x: f32) -> f32 { SS_FSCALE * (SS_XSCALE * x).tanh() }
fn activate_relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0} }

fn grad_linear(_x: f32, _f: f32) -> f32 { 1.0 }
fn grad_logistic(_x: f32, f: f32) -> f32 {
    f * (1.0 - f)
}
fn grad_ss(_x: f32, f: f32) -> f32 {
    SS_XSCALE * (SS_FSCALE - f * f/ SS_FSCALE)
}
fn grad_relu(x: f32, _f: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

impl ActivationFunction {
    fn af(&self) -> (fn(f32) -> f32) {
        use self::ActivationFunction::*;
        match *self {
            Linear => activate_linear,
            Sigmoid => activate_logistic,
            SymmetricSigmoid => activate_ss,
            ReLU => activate_relu
        }
    }

    fn agf(&self) -> (fn(f32, f32) -> f32) {
        use self::ActivationFunction::*;
        match *self {
            Linear => grad_linear,
            Sigmoid => grad_logistic,
            SymmetricSigmoid => grad_ss,
            ReLU => grad_relu
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerDesc {
    /// number of inputs, not includes bias
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub activation: ActivationFunction
}

impl LayerDesc {
    pub fn new(n_in: usize, n_out: usize, f: ActivationFunction) -> LayerDesc {
        LayerDesc {num_inputs: n_in, num_outputs: n_out, activation: f }
    }

    // total number of weights activated
    // fn num_weights(self) -> usize {
    //     (self.num_inputs + 1) * self.num_outputs
    // }
}

#[derive(Debug, Clone)]
struct Layer {
    m: Array2<f32>,
    bias: Array1<f32>,
    act: ActivationFunction,

    dm: Array2<f32>,
    dbias: Array1<f32>,

    //x: Array1<f32>
}

/// take the outer-product of a and b, applying it to c
fn outer_product<Ta, Tb, Tc>(c: &mut ArrayBase<Tc, Ix2>,
                             a: &ArrayBase<Ta, Ix1>,
                             b: &ArrayBase<Tb, Ix1>)
    where Ta: Data<Elem=f32>, Tb: Data<Elem=f32>, Tc: DataMut<Elem=f32> {
    for (mut row, ai) in izip!(c.outer_iter_mut(), a) {
        assert_eq!(row.dim(), b.dim());
        Zip::from(&mut row).and(b).apply(|r, bi| *r = *ai * *bi);
    }
}

// Quick matrix-vector multiplication
// c = A * b
#[allow(unused)]
fn mat_vec_mul<Ta, Tb, Tc>(c: &mut ArrayBase<Tc, Ix1>,
                           a: &ArrayBase<Ta, Ix2>,
                           b: &ArrayBase<Tb, Ix1>)
    where Ta: Data<Elem=f32>, Tb: Data<Elem=f32>, Tc: DataMut<Elem=f32> {
    for (ci, ar) in izip!(c, a.outer_iter()) {
        *ci = ar.dot(b);
    }
}

// c = A^t * b
fn mat_t_vec_mul<Ta, Tb, Tc>(c: &mut ArrayBase<Tc, Ix1>,
                           a: &ArrayBase<Ta, Ix2>,
                           b: &ArrayBase<Tb, Ix1>)
    where Ta: Data<Elem=f32>, Tb: Data<Elem=f32>, Tc: DataMut<Elem=f32> {

    for (ci, ar) in izip!(c, a.axis_iter(Axis(1))) {
        *ci = ar.dot(b);
    }
}

impl Layer {
    pub fn from_desc(desc: &LayerDesc) -> Layer {
        let m = Array::random((desc.num_outputs, desc.num_inputs), Range::new(-0.01, 0.01));
        let bias = Array::zeros( desc.num_outputs );
        //let x = Array::zeros( desc.num_outputs );

        let dm = Array::zeros((desc.num_outputs, desc.num_inputs));
        let dbias = Array::zeros(desc.num_outputs);

        Layer { m, bias, //x,
                dm, dbias, act: desc.activation }
    }

    pub fn num_inputs(&self) -> usize {
        self.m.dim().1
    }

    pub fn num_outputs(&self) -> usize {
        self.m.dim().0
    }

    /// Evaulate input, placing the result into output.
    pub fn evaluate_onto<T1, T2>(&self, input: &ArrayBase<T1, Ix1>,
                                 output: &mut ArrayBase<T2, Ix1>)

        where T1: Data<Elem=f32>, T2: DataMut<Elem=f32> {
        let f = self.act.af();
        for (a, r, b) in izip!(output.iter_mut(), self.m.outer_iter(), &self.bias) {
            *a = f(r.dot(input) + b);
        }
    }

    pub fn evaluate<T1>(&self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem=f32> {
        let mut arr = Array::zeros(self.bias.dim());
        self.evaluate_onto(input, &mut arr);
        arr
    }

    pub fn evaluate_onto_partial_g<T1, T2>(&mut self,
                                           input: &ArrayBase<T1, Ix1>,
                                           output: &mut ArrayBase<T2, Ix1>)

        where T1: Data<Elem=f32>, T2: DataMut<Elem=f32> {
        let f = self.act.af();
        let g = self.act.agf();

        for (a, r, b, x) in izip!(output.iter_mut(), self.m.outer_iter(), &self.bias, &mut self.dbias) {
            let pa = r.dot(input) + b;
            *a = f(pa);
            *x = g(pa, *a);
        }

        // compute the gradient of the weights, with respect to the outputs
        outer_product(&mut self.dm, &self.dbias, input);
    }

    pub fn evaluate_partial_g<T1>(&mut self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem=f32> {
        let mut arr = Array::zeros(self.bias.dim());
        self.evaluate_onto_partial_g(input, &mut arr);
        arr
    }

    /// Complete the evaluation of the gradient, taking in the
    /// gradient with respect to the outputs.
    fn complete_g<T: Data<Elem=f32>>(&mut self, dout: &ArrayBase<T, Ix1>) -> Array1<f32> {
        assert_eq!(dout.dim(), self.num_outputs());

        // We assume that we previously called evaluate*_partial_g. Now, we finish.
        // Finish gradient w/rt weight
        for (mut row, di) in self.dm.outer_iter_mut().zip(dout) {
            for x in row.iter_mut() {
                *x *= *di;
            }
        }

        // Finish gradient w/rt bias
        Zip::from(&mut self.dbias).and(dout).apply(|a, b| *a *= *b);

        let mut din = Array::zeros(self.num_inputs());
        mat_t_vec_mul(&mut din, &self.dm, dout);
        din
    }

    /// Apply the gradient
    fn gradient_step(&mut self, rate: f32) {
        Zip::from(&mut self.m).and(&self.dm).apply(|a, da| *a += da * rate);
        Zip::from(&mut self.bias).and(&self.dbias).apply(|a, da| *a += da * rate);
    }
}

pub struct NeuralNet {
    layers: Vec<Layer>
}

impl NeuralNet {
    pub fn new(layers: &[LayerDesc]) -> Option<NeuralNet>  {
        // make sure the layers are valid
        if layers.iter().tuple_windows::<(_,_)>()
            .any(|(d1, d2)| { d1.num_outputs == d2.num_inputs }) {
                return None;
            }

        Some(NeuralNet { layers: layers.iter().map(Layer::from_desc).collect() })
    }

    // Dimensions of the input
    pub fn num_inputs(&self) -> usize {
        self.layers[0].num_inputs()
    }

    // Dimensions of the output
    pub fn num_outputs(&self) -> usize {
        self.layers[self.layers.len() - 1].num_outputs()
    }

    /// Feed the input forward through the neural networks.
    pub fn evaluate<T1>(&self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem=f32> {
        assert_eq!(input.dim(), self.layers[0].num_inputs());
        self.layers.iter().fold(input.to_owned(), |x, layer| layer.evaluate(&x))
    }

    /// Evaluate, and internally store the gradient.
    pub fn evaluate_with_gradient<T1>(&mut self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem=f32> {
        assert_eq!(input.dim(), self.layers[0].num_inputs());

        let output = self.layers.iter_mut()
            .fold(input.to_owned(), |x, layer| layer.evaluate_partial_g(&x));

        let dout = Array::from_elem(self.num_outputs(), 1.0);
        self.layers.iter_mut().rev()
            .fold(dout, |x, layer| layer.complete_g(&x));

        output
    }

    /// Move all weights by a factor of alpha * grad(x)
    pub fn gradient_step(&mut self, rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.gradient_step(rate);
        }
    }
}

