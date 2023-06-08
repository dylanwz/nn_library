use std::{vec::Vec};
use ndarray::{Array, ArrayBase, OwnedRepr, Dim};
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, Rng};

// fn print_type_of<T>(_: &T) {
//     println!("{}", std::any::type_name::<T>())
// }

#[allow(dead_code)]
pub struct NNetwork {
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub num_hidden_layers: u32,
    pub weights: Vec<Array<f32, Dim<[usize; 2]>>>,
    pub biases: Vec<Vec<f32>>,
    pub ovalues: Vec<f32>,
}

// Generates the number of neurons in the (singular) hidden layer for now
fn get_hidden_neurons(ni: u32, no: u32, nhl: u32) -> Vec<u32> {
    let mut v = vec![ni];
    v.append(&mut vec![(ni + no)/2; nhl as usize]);
    v.push(no);

    return v;
}

impl NNetwork {
    /// Initialises an instance of a neural network, using multilayer perceptron
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
        // for each hidden layer, randomise the weights from each neuron of the previous layer to each
        // neuron of the current layer
        for i in 1..=(num_hidden_layers + 1) {
            connection_array.push(
                Array::random((neuron_nums[i as usize] as usize, neuron_nums[(i-1) as usize] as usize), Uniform::new(-1., 1.))
            );
            bias_array.push(Vec::new());
            for _j in 1..=(neuron_nums[i as usize]) {
                bias_array[(i-1) as usize].push(rng.gen_range(0.0..1.0));
            }
        }

        NNetwork {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: num_hidden_layers, // for the time being, this will be 1
            weights: connection_array,
            biases: bias_array,
            ovalues: vec![0.0; num_outputs as usize],
        }
    }

    /// Feed the input parameters forwards, through the network and its weights and biases
    ///     Inputs:     instance of NN, input vector
    ///     Outputs:    none... updates the 'output' field of the 'ovalues' field of a given NN
    ///     Note:       
    pub fn feed_forward(nn: &mut NNetwork, i: Vec<f32>) -> () {
        let mut v: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_vec(i).into_shape((nn.num_inputs as usize,1)).unwrap();
        for i in 0..=nn.num_hidden_layers {
            let m = &nn.weights[i as usize];
            v = m.dot(&v) + Array::from_vec(nn.biases[i as usize].clone()).into_shape((nn.biases[i as usize].len() as usize, 1)).unwrap();
        };
        for o in 0..nn.num_outputs {
            nn.ovalues[o as usize] = v[[o as usize, 0 as usize]];
        }
    }

    /// Computes the cost of the sample
    ///     Inputs:     instance of a training sample, instance of a NN
    ///     Outputs:    float (cost)
    ///     Note:       uses a cost function: (x - y)^2
    pub fn get_cost(&self) -> () {
        return;
    }

    /// Propagates the cost function backwards to compute the value of the descent gradient
    ///     Inputs:     instance of a NN, cost
    ///     Outputs:    descent gradient
    ///     Note:       
    pub fn backprop(&self) -> () {
        return;
    }

    /// Updates the parameters based on the descent gradient
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
        assert_eq!(1, 1);
    }

    #[test]
    fn feed_forward() {
        let mut nn: NNetwork = NNetwork::init(4, 6, 1);
        NNetwork::feed_forward(&mut nn, vec![0.0, 1.0, 0.0, 0.0]);
        println!("{:?}", nn.ovalues);
    }
}
