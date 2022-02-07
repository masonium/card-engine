use crate::learning::neural_net::NeuralNet;
use ndarray::prelude::*;
use ndarray::Data;

pub enum LearningModelError {
    MismatchedSize,
}

/// Reinforcement Learning traits and implementations
pub trait LearningModel {
    // Evaluate the gradient
    fn evaluate_q(&self, p: &ArrayView<f32, Ix1>) -> f32;

    // Compute q, and the gradient.
    fn evaluate_q_grad(&self, p: &ArrayView<f32, Ix1>, grad: ArrayViewMut<f32, Ix1>) -> f32;

    // input size
    fn input_size(&self) -> usize;

    // num parameters
    fn num_parameters(&self) -> usize;

    // Update the weights of the model
    fn update_weights<T: Data<Elem = f32>>(&mut self, error: f32, dir: &ArrayBase<T, Ix1>);
}

impl LearningModel for NeuralNet {
    fn evaluate_q(&self, view: &ArrayView<f32, Ix1>) -> f32 {
        self.evaluate(view)[0]
    }

    fn evaluate_q_grad(&self, p: &ArrayView<f32, Ix1>, grad: ArrayViewMut<f32, Ix1>) -> f32 {
        self.evaluate_with_gradient(p, grad)[0]
    }

    fn input_size(&self) -> usize {
        self.num_inputs()
    }

    fn num_parameters(&self) -> usize {
        self.num_parameters()
    }

    fn update_weights<T: Data<Elem = f32>>(&mut self, error: f32, dir: &ArrayBase<T, Ix1>) {
        self.update_weights(error, dir.view())
    }
}
