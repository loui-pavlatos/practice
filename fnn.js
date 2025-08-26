
let input;
let hidden = {};
let output;
let weights = {}
let biases = {}
let activations = {}
let prediction;
let loss;
let target;
function relu(i) {
    return Math.max(0, i);
}
function mse(prediction, target) {
    return .5 * ((prediction - target) * (prediction - target));
}
function dot(matrix1, matrix2) {
    let total = 0;
    for (let i = 0; i < matrix1.length; i++) {
        total += matrix1[i] * matrix2[i]
    }
    return total;
}
/*
function to calculate the gradient of the loss function with respect to a 
weight.


 */
function lossGradient(layerKey, neuronIndex, weightIndex) {
    const dlDpred = prediction - target;
    
    let gradient = dlDpred;
    const layers = Object.keys(hidden);
    const targetLayerIndex = layers.indexOf(layerKey);
    
    for (let i = layers.length - 1; i > targetLayerIndex; i--) {
        const currentLayer = layers[i];
        let layerGradient = 0;
        
        for (let j = 0; j < hidden[currentLayer].length; j++) {
            const activation = hidden[currentLayer][j];
            const reluDerivative = activation > 0 ? 1 : 0;
            layerGradient += gradient * weights[currentLayer][j][neuronIndex] * reluDerivative;
        }
        gradient = layerGradient;
    }
    
    const prevLayerKey = targetLayerIndex > 0 ? layers[targetLayerIndex - 1] : 'h0';
    const inputValue = hidden[prevLayerKey][weightIndex];
    const currentActivation = hidden[layerKey][neuronIndex];
    const reluDerivative = currentActivation > 0 ? 1 : 0;
    
    return gradient * inputValue * reluDerivative;
}

/*
inputNodes = number of input inputNodes
hidden = array. for example, hidden layer 1 with 4 neurons, hidden layer 2 with 5 neurons: [4, 5]
outputNodes = number of output nodes
 */
function init(layers) {

    hidden = {};
    hidden['h0'] = new Array(layers[0]).fill(0)
    for (let i = 1; i < layers.length; i++) {
        hidden['h' + (i)] = []
        weights['h' + i] = []
        biases['h' + i] = []
        activations['h' + i] = 'relu'

        for (let j = 0; j < layers[i]; j++) {
            hidden['h' + i].push(0)
            biases['h' + i].push(0)
            weights['h' + i].push([])
            let numberOfWeights = 0
            numberOfWeights = hidden['h' + (i - 1)].length;

            for (let k = 0; k < numberOfWeights; k++) {
                weights['h' + i][j].push(Math.random() * Math.sqrt(2 / numberOfWeights))
            }
        }
    }
    //console.log(hidden)
    //console.log(output)
    //console.log(weights)
    //console.log(activations)
    //console.log(biases)
    console.log('Network Initialized')
}

function compute(inputs) {
    // compute values of first hidden layer 
    hidden['h0'] = inputs
    for (const layer of Object.keys(hidden).slice(1)) {
        for (let i = 0; i < hidden[layer].length; i++) { // loop through and calculate the linear combination for every neuron in hidden layer 'layer'
            switch (activations[layer]) {
                case 'relu': hidden[layer][i] = relu(dot(hidden['h' + (layer.slice(1) - 1)], weights[layer][i]) + biases[layer][i]); break;
            }
        }
    }
    console.log(hidden)
}
function computeError(dataPoint, label) {
    compute(dataPoint) // define input nodes and compute network prediction
    target = label;
    return mse(hidden[Object.keys(hidden).pop()], label)
}

// initialize dataset
let data = [] // inputs
let labels = [] // labeled outputs
for (let i = 0; i < 100; i++) {
    data.push([i])
    labels.push([i * 3])
}

init([1, 4, 5, 3, 1]) // initialize network. specify how many nodes in each layer