use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use itertools::{Itertools};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFunction {
    Linear,
    Logistic
}

fn activate_linear(f: f32) -> f32 { f }
fn activate_logistic(f: f32) -> f32 { 1.0 / (1.0 + (-f).exp()) }

impl OutputFunction {
    fn af(&self) -> (fn(f32) -> f32) {
        use self::OutputFunction::*;
        match *self {
            Linear => activate_linear,
            Logistic => activate_logistic
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerDesc {
    /// number of inputs, not includes bias
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub activation: OutputFunction
}

impl LayerDesc {
    pub fn new(n_in: usize, n_out: usize, f: OutputFunction) -> LayerDesc {
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
    act: OutputFunction
}

impl Layer {
    pub fn from_desc(desc: &LayerDesc) -> Layer {
        let m = Array::random((desc.num_outputs, desc.num_inputs), Range::new(-0.01, 0.01));
        let bias = Array::zeros( desc.num_outputs );

        Layer { m, bias, act: desc.activation }
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
        for ((a, r), b) in output.iter_mut().zip(self.m.outer_iter()).zip(&self.bias) {
            *a = f(r.dot(input) + b);
        }
    }

    pub fn evaluate<T1>(&self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem=f32> {
        let mut arr = Array::zeros(self.bias.dim());
        self.evaluate_onto(input, &mut arr);
        arr
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

    pub fn gradient_descent_step<T1>(&self, input: &ArrayBase<T1, Ix1>, output: &ArrayBase<T1, Ix1>, rate: f32) -> f32
        where T1: Data<Elem=f32> {
        0.0
    }
}

