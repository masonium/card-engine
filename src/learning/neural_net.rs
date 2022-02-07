use ndarray::prelude::*;
use ndarray::{Data, DataMut, Zip};
use ndarray_rand::{F32, RandomExt};
use rand::distributions;
use itertools::{Itertools, izip};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    SymmetricSigmoid,
    ReLU,
    Exp
}

fn activate_exponential(x: f32) -> f32 {
    x.exp()
}
fn activate_linear(x: f32) -> f32 {
    x
}
fn activate_logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const SS_FSCALE: f32 = 1.7159;
const SS_XSCALE: f32 = 2.0 / 3.0;
fn activate_ss(x: f32) -> f32 {
    SS_FSCALE * (SS_XSCALE * x).tanh()
}
fn activate_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn grad_linear(_x: f32, _f: f32) -> f32 {
    1.0

}
fn grad_exponential(_x: f32, f: f32) -> f32 {
    f
}

fn grad_logistic(_x: f32, f: f32) -> f32 {
    f * (1.0 - f)
}
fn grad_ss(_x: f32, f: f32) -> f32 {
    SS_XSCALE * (SS_FSCALE - f * f / SS_FSCALE)
}
fn grad_relu(x: f32, _f: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

impl ActivationFunction {
    pub fn af(&self) -> fn(f32) -> f32 {
        use self::ActivationFunction::*;
        match *self {
            Linear => activate_linear,
            Sigmoid => activate_logistic,
            SymmetricSigmoid => activate_ss,
            ReLU => activate_relu,
            Exp => activate_exponential,
        }
    }

    pub fn agf(&self) -> fn(f32, f32) -> f32 {
        use self::ActivationFunction::*;
        match *self {
            Linear => grad_linear,
            Sigmoid => grad_logistic,
            SymmetricSigmoid => grad_ss,
            ReLU => grad_relu,
            Exp => grad_exponential
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerDesc {
    /// number of inputs, not includes bias
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub activation: ActivationFunction,
}

impl LayerDesc {
    pub fn new(n_in: usize, n_out: usize, f: ActivationFunction) -> LayerDesc {
        LayerDesc {
            num_inputs: n_in,
            num_outputs: n_out,
            activation: f,
        }
    }
}

/// Single perceptron layer in a neural network.
#[derive(Debug, Clone)]
struct Layer {
    m: Array2<f32>,
    bias: Array1<f32>,
    act: ActivationFunction,
}

/// take the outer-product of a and b, applying it to c
pub fn outer_product<Ta, Tb, Tc>(c: &mut ArrayBase<Tc, Ix2>,
                             a: &ArrayBase<Ta, Ix1>,
                             b: &ArrayBase<Tb, Ix1>)
    where Ta: Data<Elem = f32>,
          Tb: Data<Elem = f32>,
          Tc: DataMut<Elem = f32>
{
    for (mut row, ai) in izip!(c.outer_iter_mut(), a) {
        assert_eq!(row.dim(), b.dim());
        Zip::from(&mut row).and(b).apply(|r, bi| *r = *ai * *bi);
    }
}

// Quick matrix-vector multiplication
// c = A * b
pub fn mat_vec_mul<Ta, Tb, Tc>(c: &mut ArrayBase<Tc, Ix1>,
                           a: &ArrayBase<Ta, Ix2>,
                           b: &ArrayBase<Tb, Ix1>)
    where Ta: Data<Elem = f32>,
          Tb: Data<Elem = f32>,
          Tc: DataMut<Elem = f32>
{
    assert_eq!(a.dim().0, c.dim());
    assert_eq!(a.dim().1, b.dim());
    
    for (ci, ar) in izip!(c, a.outer_iter()) {
        *ci = ar.dot(b);
    }
}

// c = A^t * b
fn mat_t_vec_mul<Ta, Tb, Tc>(c: &mut ArrayBase<Tc, Ix1>,
                             a: &ArrayBase<Ta, Ix2>,
                             b: &ArrayBase<Tb, Ix1>)
    where Ta: Data<Elem = f32>,
          Tb: Data<Elem = f32>,
          Tc: DataMut<Elem = f32>
{
    assert_eq!(a.dim().0, b.dim());
    assert_eq!(a.dim().1, c.dim());

    for (ci, ar) in izip!(c, a.axis_iter(Axis(1))) {
        *ci = ar.dot(b);
    }
}

impl Layer {
    pub fn from_desc(desc: &LayerDesc) -> Layer {
        let std = (desc.num_outputs as f64).sqrt();
        let m = Array::random((desc.num_outputs, desc.num_inputs),
                              F32(distributions::Normal::new(0.0, std)));
        // let m = Array::linspace(0.0, 1.0, desc.num_inputs * desc.num_outputs)
        //     .into_shape((desc.num_outputs, desc.num_inputs)).unwrap();
        let bias = Array::zeros(desc.num_outputs);

        Layer {
            m,
            bias,
            act: desc.activation,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.m.dim().1
    }

    pub fn num_outputs(&self) -> usize {
        self.m.dim().0
    }

    pub fn num_parameters(&self) -> usize {
        self.m.dim().0 * (self.m.dim().1 + 1)
    }

    pub fn l1(&self) -> f32 {
        self.m.iter().map(|x| x.abs()).sum::<f32>() + self.bias.iter().map(|x| x.abs()).sum::<f32>()
    }

    /// Evaulate input, placing the result into output.
    pub fn evaluate_onto<T1, T2>(&self,
                                 input: &ArrayBase<T1, Ix1>,
                                 output: &mut ArrayBase<T2, Ix1>)
        where T1: Data<Elem = f32>,
              T2: DataMut<Elem = f32>
    {
        assert!(input.dim() == self.num_inputs());
        let f = self.act.af();
        for (a, r, b) in izip!(output.iter_mut(), self.m.outer_iter(), &self.bias) {
            *a = f(r.dot(input) + b);
        }
    }

    pub fn evaluate<T1>(&self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem = f32>
    {
        let mut arr = Array::zeros(self.bias.dim());
        self.evaluate_onto(input, &mut arr);
        arr
    }

    pub fn evaluate_onto_partial_g<T1, T2>(&self,
                                           input: &ArrayBase<T1, Ix1>,
                                           output: &mut ArrayBase<T2, Ix1>,
                                           partial_g: ArrayViewMut<f32, Ix1>)
        where T1: Data<Elem = f32>,
              T2: DataMut<Elem = f32>
    {
        let f = self.act.af();
        let g = self.act.agf();

        let (dml, mut dbias) = partial_g.split_at(Axis(0), self.m.len());
        let mut dm = dml.into_shape(self.m.dim()).expect("must match.");
        for (a, r, b, x) in izip!(output.iter_mut(),
                                  self.m.outer_iter(),
                                  &self.bias,
                                  &mut dbias) {
            assert_eq!(r.dim(), input.dim());
            let pa = r.dot(input) + b;
            *a = f(pa);
            *x = g(pa, *a); 
        }

        // compute the gradient of the weights, with respect to the outputs
        outer_product(&mut dm, &dbias, input); // theta'(pa) * input
    }

    pub fn evaluate_partial_g<T1>(&self,
                                  input: &ArrayBase<T1, Ix1>,
                                  partial_g: ArrayViewMut<f32, Ix1>)
                                  -> Array1<f32>
        where T1: Data<Elem = f32>
    {
        let mut arr = Array::zeros(self.bias.dim());
        self.evaluate_onto_partial_g(input, &mut arr, partial_g);
        arr
    }

    /// Complete the evaluation of the gradient, taking in the
    /// gradient with respect to the outputs.
    fn complete_g<T: Data<Elem = f32>>(&self,
                                       de_dout: &ArrayBase<T, Ix1>,
                                       g: ArrayViewMut<f32, Ix1>)
                                       -> Array1<f32> {
        assert_eq!(de_dout.dim(), self.num_outputs());
        
        let (dml, mut dout_dbias) = g.split_at(Axis(0), self.m.len());
        let mut dout_dm = dml.into_shape(self.m.dim()).expect("must match.");

        // We assume that we previously called evaluate*_partial_g. Now, we finish.
        // Finish gradient w/rt weight
        for (mut row, di) in dout_dm.outer_iter_mut().zip(de_dout) {
            for x in row.iter_mut() {
                *x *= *di;
            }
        }

        // Finish gradient w/rt bias
        dout_dbias *= de_dout;

        let mut de_din = Array::zeros(self.num_inputs());
        mat_t_vec_mul(&mut de_din, &self.m, de_dout);

        de_din
    }

    /// Apply the gradient
    fn weight_step(&mut self, rate: f32, weights: ArrayView<f32, Ix1>) {
        let (dm_linear, dbias) = weights.split_at(Axis(0), self.num_inputs() * self.num_outputs());
        let dm = dm_linear
            .into_shape((self.num_outputs(), self.num_inputs()))
            .expect("gradient size must match.");
        Zip::from(&mut self.m)
            .and(&dm)
            .apply(|a, da| *a += da * rate);
        Zip::from(&mut self.bias)
            .and(dbias)
            .apply(|a, da| *a += da * rate);
    }


    #[allow(unused)]
    pub fn dump(&self) {
        println!("W:\n{}\nb\n{}\n",self.m, self.bias);
    }

    fn weights(&self, v: ArrayViewMut<f32, Ix1>) {
        assert!(v.dim() == self.num_parameters());
        let (mut ml, mut bias) = v.split_at(Axis(0), self.m.len());

        ml.assign(&ArrayView::from_shape(self.m.len(), self.m.as_slice().unwrap()).expect("must match"));
        bias.assign(&self.bias);
    }

}

#[derive(Clone)]
pub struct NeuralNetworkParameters {
    pub learning_rate: f32,
}

/// Neural network
#[derive(Clone)]
pub struct NeuralNet {
    layers: Vec<Layer>,
    param: NeuralNetworkParameters,
    current_rate: f32,
    ni: usize,
}

impl NeuralNet {
    pub fn new(layers: &[LayerDesc], lr: f32) -> Option<NeuralNet> {
        // make sure the layers are valid
        if layers
               .iter()
               .tuple_windows::<(_, _)>()
               .any(|(d1, d2)| d1.num_outputs != d2.num_inputs) {
            return None;
        }

        Some(NeuralNet {
            layers: layers.iter().map(Layer::from_desc).collect(),
            current_rate: lr, ni: 0, 
            param: NeuralNetworkParameters { learning_rate: lr },
             })
    }

    pub fn split_at(mut self, n: usize) -> (NeuralNet, NeuralNet) {
        let b = self.layers.split_off(n);
        (NeuralNet { layers: self.layers, param: self.param.clone(), 
                     current_rate: self.current_rate, ni: self.ni },
         NeuralNet { layers: b, param: self.param, current_rate: self.current_rate,
                     ni: self.ni})
    }

    // Dimensions of the input
    pub fn num_inputs(&self) -> usize {
        self.layers[0].num_inputs()
    }

    // Dimensions of the output
    pub fn num_outputs(&self) -> usize {
        self.layers[self.layers.len() - 1].num_outputs()
    }

    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }

    /// Flattened version of all weights, by layer
    pub fn weights(&self) -> Array<f32, Ix1> {
        let mut arr = Array::zeros(self.num_parameters());
        self.layers.iter().fold(arr.view_mut(), |view, layer| {
            let (a, b) = view.split_at(Axis(0), layer.num_parameters());
            layer.weights(a);
            b
        });
        arr
    }

    // pub fn dump(&self)  {
    //     for l in &self.layers {
    //         l.dump();
    //     }
    // }

    pub fn l1(&self) -> f32 {
        self.layers.iter().map(|layer| layer.l1()).sum()
    }

    /// Feed the input forward through the neural networks.
    pub fn evaluate<T1>(&self, input: &ArrayBase<T1, Ix1>) -> Array1<f32>
        where T1: Data<Elem = f32>
    {
        assert_eq!(input.dim(), self.layers[0].num_inputs());
        self.layers.iter().fold(input.to_owned(), |x, layer| { 
            let y = layer.evaluate(&x);
            y })
    }

    /// Evaluate, and internally store the gradient.
    pub fn evaluate_with_gradient<T1>(&self,
                                      input: &ArrayBase<T1, Ix1>,
                                      mut gradient: ArrayViewMut<f32, Ix1>)
                                      -> Array1<f32>
        where T1: Data<Elem = f32>
    {
        assert_eq!(input.dim(), self.layers[0].num_inputs());
        gradient.fill(0.0);
        
        let output = self.layers
            .iter()
            .fold((input.to_owned(), gradient.view_mut()), |(x, gv), layer| {
                let (g, ogv) = gv.split_at(Axis(0), layer.num_parameters());
                (layer.evaluate_partial_g(&x, g), ogv)
            }).0;

        //println!("half-way: {:8.5}", gradient);

        let de_dout = Array::from_elem(self.num_outputs(), 1.0);
        self.layers
            .iter()
            .rev()
            .fold((de_dout, gradient), |(x, gv), layer| {
                let split_loc = gv.len() - layer.num_parameters();
                let (ogv, g) = gv.split_at(Axis(0), split_loc);
                (layer.complete_g(&x, g), ogv)
            });

        output
    }

    /// Move all weights by a factor of alpha * e * grad(x)
    pub fn update_weights(&mut self, err: f32, w: ArrayView<f32, Ix1>) {
        self.ni += 1;
        let lre = self.param.learning_rate * err * 1.0 / (1.0 + 0.001 * (self.ni as f32));
        self.layers
            .iter_mut()
            .fold(w, |weights, layer| {
                let (g, mw) = weights.split_at(Axis(0), layer.num_parameters());
                layer.weight_step(lre, g);
                mw
            });
    }
}


#[cfg(test)]
mod tests {
    use super::ActivationFunction;
    use ndarray::Array;

    #[test]
    fn test_linear() {
        test_grad(ActivationFunction::Linear);
    }

    #[test]
    fn test_relu() {
        test_grad(ActivationFunction::ReLU);
    }

    #[test]
    fn test_sigmoid() {
        test_grad(ActivationFunction::Sigmoid);
    }


    #[test]
    fn test_sym_sigmoid() {
        test_grad(ActivationFunction::SymmetricSigmoid);
    }

    fn test_grad(af: ActivationFunction) {
        let f = af.af();
        let g = af.agf();
        for r in Array::linspace(-1.95, 1.95, 40).iter() {
            let g_est = (f(r + 5e-4) - f(r - 5e-4)) / 1e-3;
            let g_actual = g(*r, f(*r));

            println!("{} vs. {}", g_est, g_actual);
            assert!((g_est - g_actual).abs() < 1e-3);
        }
    }


}
