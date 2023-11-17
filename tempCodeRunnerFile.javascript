function printFibonacciSeries(totalCount) {
    let fib = [];

    fib[0] = 0;

    if (totalCount > 0) {
        fib[1] = 1;
        console.log(fib[0]);
        if (totalCount > 1) {
            console.log(fib[1]);
        }
    }

    for (let i = 2; i < totalCount; i++) {
        fib[i] = fib[i - 1] + fib[i - 2];
        console.log(fib[i]);
    }
}

// Example: To print the Fibonacci series up to 20 numbers
printFibonacciSeries(20);

