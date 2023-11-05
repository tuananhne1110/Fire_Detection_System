<?php

$url = 'http://localhost:5000/signin';  // Update with your Flask app's URL
$data = [
    'username' => 'admin',
    'password' => 'password123',
];

$options = [
    'http' => [
        'method' => 'POST',
        'header' => 'Content-Type: application/x-www-form-urlencoded',
        'content' => http_build_query($data),
    ],
];

$context = stream_context_create($options);
$response = file_get_contents($url, false, $context);

if ($response) {
    $result = json_decode($response, true);
    if (isset($result['success'])) {
        echo 'Login successful: ' . $result['success'];
    } else {
        echo 'Login failed: ' . $result['error'];
    }
} else {
    echo 'Error connecting to the Flask app';
}
