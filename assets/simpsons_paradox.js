var data = {alg1: [0.8, 0.2], alg2: [0.75, 0.75]};
var performance = function(x) {
    var alg1 = data.alg1[0] * x/100 + data.alg1[1] * (100-x)/100;
    var alg2 = data.alg2[0] * x/100 + data.alg2[1] * (100-x)/100;
    return [alg1, alg2]
};

var perfData = performance(95);
document.getElementById("simpsonsMPText").textContent = "Percentage of Major Publisher content in logs: " + Number(95.0) + "%.";

var updatePerfVals = function(value) {
    document.getElementById("simpsonsMPText").textContent = "Percentage of Major Publisher content in logs: " + Number(value) + "%.";
    perfData = performance(value);
    chartData.datasets[0].data = [perfData[0], perfData[1]];
    window.simpsonChart.update();
};

var colour = Chart.helpers.color;
var chartData = {
    labels: ['Algorithm 1', 'Algorithm 2'],
    datasets: [{
        label: 'Algorithm 1',
        backgroundColor: [
            colour(window.chartColours.red).alpha(0.5).rgbString(),
            colour(window.chartColours.blue).alpha(0.5).rgbString()
        ],
        borderColor: [
            window.chartColours.red,
            window.chartColours.blue
        ],
        borderWidth: 1.5,
        data: [perfData[0], perfData[1]]
    }]
};

window.onload = function() {
    var ctx = document.getElementById("simpsons_canvas").getContext('2d');
    window.simpsonChart = new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            legend: {
                display: false,
                position: 'top',
            },
            title: {
                display: true,
                text: 'Simpson\'s paradox'
            },
            scales: {
                xAxes: [{
                    barPercentage: 0.70
                }],
                yAxes: [{
                    ticks: {
                        min: 0.6
                    }
                }]
            }
        }
    });
};