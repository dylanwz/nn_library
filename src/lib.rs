use std::vec::Vec;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub struct NNetwork {
    num_inputs: f32,
    num_outputs: f32,
    num_hidden_layers: u32,
    connections: Vec<Vec<Connection>>,
    iovalues: Vec<Vec<f32>>,
}

struct Connection {
    weight: f32,
    bias: f32,
}

impl NNetwork {
    /// Feed the input parameters forwards, through the network and its weights and biases
    ///     Inputs:     input size, output size, number of hidden layers
    ///     Outputs:    a neural network struct, with a 2D array of edges and biases
    ///     Note:       
    pub fn new(inputs: f32, outputs: f32, num_hidden_layers: u32) -> NNetwork {
        return NNetwork {
            num_inputs: inputs,
            num_outputs: outputs,
            num_hidden_layers: num_hidden_layers,
            // randomise this LOLLL!!
            connections: vec![vec![Connection {weight: 0.0, bias: 0.0}]],
            iovalues: vec![vec![0.0]],

        };
    }

    /// Initialise the weights and biases of a NN
    ///     Inputs:     instance of NN
    ///     Outputs:    none... updates 'connections' field of a given NN
    ///     Note:       
    pub fn init_params(&self) -> [u32; 2] {
        return [1, 2];
    }

    /// Feed the input parameters forwards, through the network and its weights and biases
    ///     Inputs:     instance of NN
    ///     Outputs:    none... updates the 'output' field of the 'iovalues' field of a given NN
    ///     Note:       
    pub fn feed_forward(&self) -> () {
        return;
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
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
