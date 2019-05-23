function AliasTable(counts) {   
    var n = counts.length;
    var norm = counts.reduce(function(a, b) { return a + b; }, 0);

    this.prob = new Array(n);
    this.alias = new Array(n);
    this.stop = false;

    // divide into a rich/poor list
    var small = [], large = [], p = new Array(n);
    for (var i = 0; i < n; i += 1) {
        p[i] = n * counts[i] / norm;
        if (p[i] < 1) {
            small.push(i);
        } else {
            large.push(i);
        }

        this.prob[i]  = p[i];
    }

    /// build up a list of transitions;
    this.history = [];

    // hack: include initial state 3x so it lingers
    for (i = 0; i < 2; ++i) {
        this.history.push({'alias' : this.alias.slice(),
                         'p' : p.slice() });
    }

    // build up alias table 
    while (small.length && large.length) {
        var l = small.pop(), g = large.pop();

        this.prob[l] = p[l];
        this.alias[l] = g;

        p[g] = (p[g] + p[l]) - 1.0;

        this.history.push({'alias' : this.alias.slice(),
                           'p' : p.slice() });
        if (p[g] < 1) {
            small.push(g);
        } else {
            large.push(g);
        }
    }

    // handle numerical instability
    while (large.length) {
        var g = large.pop();
        this.prob[g] = 1.0;
    }

    while (small.length) {
        var l = small.pop();
        this.prob[l] = 1.0;
    }
}

AliasTable.prototype.generate = function() {
    var i = Math.floor(Math.random() * (this.prob.length));
    if (Math.random() <= this.prob[i]) {
        return i;
    } else {
        return this.alias[i];
    }
};

AliasTable.prototype.displayState = function(div, state) {
    var maxWeight = Math.max.apply(null, table.history[0].p);

    var width = div.nodes()[0].offsetWidth,
        height = width * 0.5,
        cellWidth = Math.floor(width / this.prob.length);

    // create svg if not already existing
    div.selectAll("svg").data([0]).enter().append("svg");

    var svg = div.select("svg")
        .attr("width", width)
        .attr("height", height);

    var totalHeight = height;
    height /= maxWeight;

    // todo: line at total height?
    prob = this.history[state].p;
    alias = this.history[state].alias;

    var nodes = svg.selectAll(".cell")
        .data(prob);

    var enter = nodes.enter()
        .append("g")
        .attr("class", "cell");

    enter.append("rect")
        .attr("class", "prob")
        .style("fill-opacity", 0.8)
        .attr("stroke", "white")
        .attr("stroke-width", 2)
        .attr("fill", function (d, i) { return d3.schemeCategory10[i]; });

    enter.append("text")
        .attr("class", "probLabel")
        .style("font-size", "10px")
        .style("font-family", '"HelveticaNeue", "Helvetica Neue", "Segoe UI", Arial, sans-serif')
        .attr("dy", ".35em")
        .text(function(d, i) { return i; })
        .attr("fill", "white");

    enter.append("rect")
        .attr("class", "alias")
        .style("fill-opacity", 0.8)
        .attr("stroke", "white")
        .attr("stroke-width", 2);

    enter.append("text")
        .attr("class", "aliasLabel")
        .attr("dy", ".35em")
        .style("font-family", '"HelveticaNeue", "Helvetica Neue", "Segoe UI", Arial, sans-serif')
        .style("font-size", "10px")
        .attr("fill", "white");

    transition = div.transition().duration(state ?  1000 : 500);

    div.selectAll(".alias")
        .style("fill-opacity", function(d,i) { return alias[i] === undefined ? 0 : 0.8; })
        .style("stroke-opacity", function(d,i) { return alias[i] === undefined ? 0 : 0.8; });

    transition.selectAll(".prob")
        .attr("width", cellWidth)
        .attr("x", function(d, i) { return i * cellWidth; } )
        .attr("y", function(d, i) { return totalHeight - height * prob[i]; })
        .attr("height", function(d, i) { return height * prob[i]; });

    transition.selectAll(".probLabel")
        .attr("x", function(d, i) { return (i + 0.5) * cellWidth; } )
        .attr("y", function(d, i) { return (totalHeight - height * prob[i] / 2); } );
       
    transition.selectAll(".alias")
        .attr("width", cellWidth)
        .attr("x", function(d, i) { return i * cellWidth; })
        .attr("fill", function (d, i) { return d3.schemeCategory10[alias[i]]; })
        .attr("y", totalHeight - height)
        .attr("height", function(d, i) { return (alias[i] === undefined) ? 0 : height * (1 - prob[i]); });

    transition.selectAll(".aliasLabel")
        .text(function(d, i) { return alias[i]; })
        .attr("x", function(d, i) { return (i + 0.5) * cellWidth; })
        .attr("y", function(d, i) { return totalHeight - height + height * (1 - prob[i]) / 2; });

    // animate it
    table = this;
    transition.on("end", function() {
        if (table.stop) {
            return;
        }
        state = state + 1;
        if (state >= table.history.length) {
            div.transition().duration(5000).on("end", function() {
                table.displayState(div ,0); 
            });
        } else {
            table.displayState(div, state);
        }
    });
};