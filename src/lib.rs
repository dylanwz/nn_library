use std::vec::Vec;

#[allow(dead_code)]
pub struct NNetwork {
    num_inputs: u32,
    num_outputs: u32,
    num_hidden_layers: u32,
    connections: Vec<Vec<Connection>>,
    iovalues: Vec<Vec<f32>>,
}

#[allow(dead_code)]
struct Connection {
    weight: f32,
    bias: f32,
}

impl NNetwork {
    /// Feed the input parameters forwards, through the network and its weights and biases
    ///     Inputs:     input size, output size, number of hidden layers
    ///     Outputs:    a neural network struct, with a 2D array of edges and biases
    ///     Note:       the connection values are initially random, and the i/o activations are 0
    pub fn new(num_inputs: u32, num_outputs: u32, num_hidden_layers: u32) -> NNetwork {
        let rng = rand::random::<f32>;
        return NNetwork {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: num_hidden_layers,
            // TO-DO: Figure out the architecture for the connections.
            connections: vec![vec![Connection {weight: rng(), bias: rng()}]],
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
    fn new_nn() {
        let nn: NNetwork = NNetwork::new(724, 10, 1);
        for row in &nn.connections {
            for column in row {
                print!("{}|{} ", column.weight, column.bias);
            }
            println!();
        }
        assert_eq!(1, 1);
    }
}
