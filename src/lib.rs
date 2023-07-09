use std::{vec::Vec};
use ndarray::{Array, ArrayBase, OwnedRepr, Dim};
use ndarray_rand::RandomExt;
use rand_distr::{Distribution, Normal};
use std::f64::consts::E;
use std::fs::File;
use csv;

// fn print_type_of<T>(_: &T) {
//     println!("{}", std::any::type_name::<T>())
// }

#[allow(dead_code)]
pub struct NNetwork {
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub num_hidden_layers: u32,
    pub num_neurons: Vec<u32>,
    pub weights: Vec<Array<f64, Dim<[usize; 2]>>>,
    pub biases: Vec<Vec<f64>>,
    pub activations: Vec<Vec<f64>>,
    pub partials: Vec<Vec<f64>>,
}

// Generates the number of neurons in the (for now, singular) hidden layer
fn get_hidden_neurons(ni: u32, no: u32, nhl: u32) -> Vec<u32> {
    let mut v = vec![ni];
    v.append(&mut vec![(ni + no)/2; nhl as usize]);
    v.push(no);

    return v;
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
        let normal = Normal::new(0.0, (2.0/((num_inputs + num_outputs) as f64)).sqrt()).unwrap();

        let neuron_nums: Vec<u32> = get_hidden_neurons(num_inputs, num_outputs, num_hidden_layers);
        let mut connection_array = Vec::new();
        let mut bias_array: Vec<Vec<f64>> = Vec::new();
        let mut activation_array: Vec<Vec<f64>> = Vec::new();
        // for each hidden layer, randomise the weights from each neuron of the previous layer to each
        // neuron of the current layer
        // using Xavier initialisation
        activation_array.push(vec![0.0; num_inputs as usize]);
        for i in 1..=(num_hidden_layers + 1) {
            connection_array.push(
                Array::random_using((neuron_nums[i as usize] as usize, neuron_nums[(i-1) as usize] as usize), normal, &mut rng)
            );
            bias_array.push(Vec::new());
            activation_array.push(Vec::new());
            for _ in 1..=(neuron_nums[i as usize]) {
                bias_array[(i-1) as usize].push(normal.sample(&mut rng) as f64);
                activation_array[i as usize].push(0.0);
            }
        }
        activation_array[(num_hidden_layers + 1) as usize] = vec![0.0; num_outputs as usize];
        let partial_array = activation_array.clone();

        NNetwork {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: num_hidden_layers,
            num_neurons: neuron_nums,
            weights: connection_array,
            biases: bias_array,
            activations: activation_array,
            partials: partial_array,
        }
    }

    /// 2. Feed the input parameters forwards, through the network and its weights and biases
    ///     Inputs:     instance of NN, input vector
    ///     Outputs:    none... updates the 'output' field of the 'ovalues' field of a given NN
    ///     Note:       
    pub fn feed_forward(&mut self, i: Vec<f64>) -> () {
        let mut v: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array::from_vec(i)
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
            .mapv_into(|x| 1.0/(1.0+(E.powf(-1.0 * x))));

            for n in 0..v.dim().0 {
                self.activations[(l+1) as usize][n] = v[[n as usize, 0 as usize]];
            };
        };
    }
                                                                             
    /// 3a. Computes the cost of the sample
    ///     Inputs:     instance of a training sample, instance of a NN
    ///     Outputs:    float (cost)
    ///     Note:       uses a cost function: (x - y)^2
    pub fn get_cost(&self, v: &Vec<f64>) -> f64 {
        fn cost_fn(a: f64, b: f64) -> f64 {
            return (1.0/2.0) * (a-b).powf(2.0); // (x-y)^2
        }
        let mut cost = 0.0;
        for e in 0..self.num_outputs {
            cost += cost_fn(self.activations[(self.num_hidden_layers + 1) as usize][e as usize], v[e as usize]);
        }
        return cost;
    }

    /// 3b. Propagates the cost function backwards to compute the value of the descent gradient
    ///     Inputs:     instance of a NN, cost
    ///     Outputs:    none; alters partial field of NN. stochastic gradient
    ///     Note:       the idea: ∂C/∂W(n)(ji) = ∂C/∂a0 * [∂a0/∂z0 * ∂z0/∂a1] * ... * [∂a(n-1)/∂z(n-1) * ∂z(n-1)/∂zn] * ∂zn/∂W(n)(ji)
    pub fn backprop(&mut self, expected_vals: &Vec<f64>) -> () {        
        let mut sum: f64;
        for layer in (0..self.partials.len()).rev() { // go layer by layer --- (!!) each layer is a 'unit' (!!)
            if layer == (self.partials.len() - 1) { // initial partial derivative: ∂C/∂W(n)(ji) = ∂C/∂a0 = 2(x-y)
                for outp in 0..self.activations[layer].len() {
                    self.partials[layer][outp] = (self.activations[layer][outp] - expected_vals[outp]) * ((E.powf(-1.0 * self.activations[layer][outp] as f64))/(1.0 + E.powf(-1.0 * self.activations[layer][outp] as f64)).powf(2.0));
                }
            } else {
                for curr in 0..self.partials[layer].len() { // summing each path of influence (!!) leading up to this neuron (!!)
                    sum = 0.0; // use the distributive law of addition and multiplication... weight changes, but ∂a(n)/∂z(n) is the same for all paths
                    for prev in 0..self.partials[layer + 1].len() {
                        sum += self.partials[layer + 1][prev] * self.weights[layer][[prev,curr]]; // ∂z(n-1)/∂a(n) = W(n), where z is the endpoint of W
                    }
                    self.partials[layer][curr] = sum *
                    ((E.powf(-1.0 * self.activations[layer][curr] as f64))/(1.0 + E.powf(-1.0 * self.activations[layer][curr] as f64)).powf(2.0)); // ∂a(n)/∂z(n)
                }
            }
        }
    }

    /// 4. Updates the parameters based on the descent gradient
    ///     Inputs:     instance of a NN, descent gradient, learning rate
    ///     Outputs:    none... updates the 'connections' field of a given NN
    ///     Note:       
    pub fn update_params(&mut self, learning_rate: f64) -> () {
        for layer in (1..self.partials.len()).rev() { // starting at the back
            for curr in 0..(self.partials[layer].len()) {
                self.biases[layer - 1][curr] -= learning_rate * self.partials[layer][curr]; // update biases... ∂zn/∂b(n)(j) = 1
                for prev in 0..(self.activations[layer - 1].len()) {
                    // ∂zn/∂W(n)(ji) = ... * a(n-1)
                    self.weights[layer - 1][[curr, prev]] -= learning_rate * self.partials[layer][curr] * self.activations[layer - 1][prev];
                }
            }
        }
        return;
    }

    pub fn csv_to_training(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let mut reader = csv::Reader::from_reader(file);

        let mut data: Vec<Vec<f64>> = Vec::new();

        for record in reader.records() {
            let record = record?;
            let mut row: Vec<f64> = Vec::new();

            for field in record.iter() {
                let value = field.parse::<f64>()?;
                row.push(value);
            }

            data.push(row);
        }

        Ok(data)
    }

    pub fn do_train(&mut self, inputs: Vec<f64>, desired_output: &Vec<f64>, alpha: f64) -> () {
        NNetwork::feed_forward(self, inputs);
        NNetwork::backprop(self, desired_output);
        NNetwork::update_params(self, alpha)
    }

    pub fn do_test(&mut self, test_inputs: Vec<f64>, label: i32) -> bool {
        NNetwork::feed_forward(self, test_inputs);
        println!("
            Received: {:?} \n
            Expected: {:?}
        ", self.activations[(self.num_hidden_layers + 1) as usize], label);
        if find_max_value(&self.activations[(self.num_hidden_layers + 1) as usize]) == (label) {
            return true;
        } else {
            return false;
        }
    }
}

fn find_max_value(values: &[f64]) -> i32 {
    let mut max_value = f64::NEG_INFINITY;
    let mut max_index = 0;
    for i in 0..values.len() {
        if values[i] > max_value {
            max_value = values[i];
            max_index = i as i32;

        }
    }

    max_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_nn() {
        let nn: NNetwork = NNetwork::init(5,3,1);
        println!("Weights: {:?}", nn.weights);
        println!("Biases: {:?}", nn.biases);
        println!("Activations: {:?}", nn.activations);
        println!("Partials: {:?}", nn.partials);
    }

    #[test]
    fn feed_forward() {
        let mut nn: NNetwork = NNetwork::init(15, 10, 2);
        NNetwork::feed_forward(&mut nn, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        println!("{:?}", nn.activations[(nn.num_hidden_layers + 1) as usize]);
    }

    #[test]
    fn backprop() {
        let mut nn: NNetwork = NNetwork::init(3, 2, 1);
        NNetwork::feed_forward(&mut nn, vec![0.0, 0.2, 0.1]);
        println!("State: A: {:?}... \n W: {:?}", nn.activations, nn.weights);
        NNetwork::backprop(&mut nn, &vec![0.0, 1.0]);
        println!("Partials: {:?}", nn.partials);
        println!("Old Weights: {:?}", nn.weights);
        println!("Old Biases: {:?}", nn.biases);        
        NNetwork::update_params(&mut nn, 0.01);
        println!("Updated Weights: {:?}", nn.weights);
        println!("Updated Biases: {:?}", nn.biases);        
    }

    #[test]
    fn csv_to_training() {
        let epoch = NNetwork::csv_to_training("./mnist_train.csv"); // index 0 is the label, index 1 to index 783 are values
        println!("{:?}", Result::unwrap(epoch)[3][1..=784].to_vec().len());
    }

    #[test]
    fn final_result() {
        let mut nn: NNetwork = NNetwork::init(784, 10, 1);
        let epoch = Result::unwrap(NNetwork::csv_to_training("./mnist_train.csv"));
        let mut c = 1;
        for ex in epoch {
            if c == 10000 {
                continue;
            }
            let mut desired_output: Vec<f64> = vec![0.0; 10];
            desired_output[ex[0] as usize] = 1.0;
            NNetwork::do_train(&mut nn, ex[1..=784].to_vec(), &desired_output, 0.1);
            println!("Trained NN on {} training sets...", c);
            c += 1;
        }
        let tests = Result::unwrap(NNetwork::csv_to_training("./mnist_test.csv"));
        let mut total = 0.0;
        let mut correct = 0.0;
        for t in tests {
            if total == 200.0 {
                continue;
            }
            let res = NNetwork::do_test(&mut nn, t[1..=784].to_vec(), t[0] as i32);
            if res == true {
                correct += 1.0;
                total += 1.0;
            } else {
                total += 1.0;
            }
        }
        // println!("WEIGHTS! {:?} \n BIASES! {:?}", nn.weights, nn.biases);
        // println!{"Hmm.. {:?}", nn.activations};
        println!("Final accuracy: {}, correct: {}", correct/total as f64, correct as f64);
    }
}