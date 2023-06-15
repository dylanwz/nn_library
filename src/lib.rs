use std::{vec::Vec};
use ndarray::{Array, ArrayBase, OwnedRepr, Dim};
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, Rng};
use std::f32::consts::E;

// fn print_type_of<T>(_: &T) {
//     println!("{}", std::any::type_name::<T>())
// }

#[allow(dead_code)]
pub struct NNetwork {
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub num_hidden_layers: u32,
    pub num_neurons: Vec<u32>,
    pub weights: Vec<Array<f32, Dim<[usize; 2]>>>,
    pub biases: Vec<Vec<f32>>,
    pub activations: Vec<Vec<f32>>,
}

// Generates the number of neurons in the (for now, singular) hidden layer
fn get_hidden_neurons(ni: u32, no: u32, nhl: u32) -> Vec<u32> {
    let mut v = vec![ni];
    v.append(&mut vec![(ni + no)/2; nhl as usize]);
    v.push(no);

    return v;
}

pub struct TrainingFeedback {
    pub weights: Vec<Array<f32, Dim<[usize; 2]>>>,
    pub biases: Vec<Vec<f32>>,
}

impl NNetwork {
    /// 1. Initialises an instance of a neural network, using multilayer perceptron
    ///     Inputs:     input size, output size, number of hidden layers
    ///     Outputs:    a neural network struct, with a 2D array of edges and biases
    ///     Note:       the connection values are initially random, and the i/o activations are 0
    ///                 weights_array: index 0 = each input neuron to each neuron of next layer
    ///                 bias_array:    index 0 = list of biases for layer 1
    pub fn init(num_inputs: u32, num_outputs: u32, num_hidden_layers: u32) -> NNetwork {
        let mut rng = rand::thread_rng();
        
        let neuron_nums: Vec<u32> = get_hidden_neurons(num_inputs, num_outputs, num_hidden_layers);
        let mut connection_array = Vec::new();
        let mut bias_array: Vec<Vec<f32>> = Vec::new();
        let mut activation_array: Vec<Vec<f32>> = Vec::new();
        // for each hidden layer, randomise the weights from each neuron of the previous layer to each
        // neuron of the current layer
        activation_array.push(Vec::new());
        for i in 1..=(num_hidden_layers + 1) {
            connection_array.push(
                Array::random((neuron_nums[i as usize] as usize, neuron_nums[(i-1) as usize] as usize), Uniform::new(-1., 1.))
            );
            bias_array.push(Vec::new());
            activation_array.push(Vec::new());
            for _j in 1..=(neuron_nums[i as usize]) {
                bias_array[(i-1) as usize].push(rng.gen_range(0.0..1.0));
                activation_array[(i-1) as usize].push(0.0);
            }
        }
        activation_array[(num_hidden_layers + 1) as usize] = vec![0.0; num_outputs as usize];

        NNetwork {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: num_hidden_layers,
            num_neurons: neuron_nums,
            weights: connection_array,
            biases: bias_array,
            activations: activation_array,
        }
    }

    /// 2. Feed the input parameters forwards, through the network and its weights and biases
    ///     Inputs:     instance of NN, input vector
    ///     Outputs:    none... updates the 'output' field of the 'ovalues' field of a given NN
    ///     Note:       
    pub fn feed_forward(&mut self, i: Vec<f32>) -> () {
        let mut v: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_vec(i)
            .into_shape((self.num_inputs as usize,1))
            .unwrap();


        for n in 0..v.dim().0 {
            self.activations[0][n] = v[[n as usize, 0 as usize]];
        };

        for l in 0..=self.num_hidden_layers {
            let m = &self.weights[l as usize];
            v = (m.dot(&v) + Array::from_vec(self.biases[l as usize].clone())
            .into_shape((self.biases[l as usize].len() as usize, 1))
            .unwrap())
            .mapv_into(|x| if x > 0.0 { 1.0/(1.0+(E.powf(-x))) } else { 0.0 });

            for n in 0..v.dim().0 {
                self.activations[(l+1) as usize][n] = v[[n as usize, 0 as usize]];
            };
        };
    }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    // /// 3a. Computes the cost of the sample
    // ///     Inputs:     instance of a training sample, instance of a NN
    // ///     Outputs:    float (cost)
    // ///     Note:       uses a cost function: (x - y)^2
    // pub fn get_cost(&self, v: &Vec<f32>) -> f32 {
    //     fn cost_fn(a: f32, b: f32) -> f32 {
    //         return (a-b).powi(2); // (x-y)^2
    //     }
    //     let mut cost = 0.0;
    //     for e in 0..self.num_outputs {
    //         cost += cost_fn(self.activations[(self.num_hidden_layers + 1) as usize][e as usize], v[e as usize]);
    //     }
    //     return cost;
    // }

    // pub fn calc_partial_w(&self, cost: f32, layer: u32, src: u32, dest: u32) -> f32 {
    //     let mut res = 2.0*cost.powi(2); // ∂C/∂W(ji) = ∂C/∂a0 ...
    //     for d in 0..=(layer) {
    //         res *= self.weights[d as usize][[,]]
    //     }
    //     return 0.0;
    // }

    // pub fn calc_partial_b(&self, depth: u32, src: u32, dest: u32) -> f32 {
    //     return 0.0;
    // }
    // /// 3b. Propagates the cost function backwards to compute the value of the descent gradient
    // ///     Inputs:     instance of a NN, cost
    // ///     Outputs:    descent gradient
    // ///     Note:       
    // pub fn backprop(&self, expected_vals: &Vec<f32>) -> TrainingFeedback {
    //     let c = NNetwork::get_cost(self, &expected_vals);
    //     let feedback = TrainingFeedback {
    //         weights: Vec::new(),
    //         biases: Vec::new(),
    //     };
    //     for layer in (self.num_hidden_layers)..0 { // start with the output layer
    //         for j in 0..(self.num_neurons[layer as usize]) { // let j be the destination neuron for a weight
                
    //             for i in (self.num_neurons[(layer - 1) as usize])..=0 {

    //             }
    //         }
    //     }
         
    //     return feedback;
    // }

    /// 4. Updates the parameters based on the descent gradient
    ///     Inputs:     instance of a NN, descent gradient, learning rate
    ///     Outputs:    none... updates the 'connections' field of a given NN
    ///     Note:       
    pub fn update_params(&self) -> () {
        return;
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_nn() {
        let nn: NNetwork = NNetwork::init(4, 6, 1);
        println!("Weights: {:?}", nn.weights);
        println!("Biases: {:?}", nn.biases);
        println!("Activations: {:?}", nn.activations);
        assert_eq!(1, 1);
    }

    #[test]
    fn feed_forward() {
        let mut nn: NNetwork = NNetwork::init(4, 6, 2);
        NNetwork::feed_forward(&mut nn, vec![0.0, 1.0, 0.0, 0.0]);
        println!("{:?}", nn.activations[(nn.num_hidden_layers + 1) as usize]);
    }

    #[test]
    fn backprop() {
        let mut nn: NNetwork = NNetwork::init(4, 6, 2);
        NNetwork::feed_forward(&mut nn, vec![0.0, 0.0, 1.0, 1.0]);
        // NNetwork::backprop(&nn, &vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
