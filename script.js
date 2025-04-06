
// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-analytics.js";

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyDiTKL2q2FPNw--fWE3Ow6wRtVryjHl7nc",
    authDomain: "altos-earthquake-detector.firebaseapp.com",
    projectId: "altos-earthquake-detector",
    storageBucket: "altos-earthquake-detector.firebasestorage.app",
    messagingSenderId: "281378243771",
    appId: "1:281378243771:web:3065cb360e395412d5959e",
    measurementId: "G-S4ECNZJSYY"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// Handle form submission
const form = document.getElementById('dataForm');
const output = document.getElementById('output');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const data = document.getElementById('dataInput').value;

    // Log the data to the console (replace this with Firebase functionality as needed)
    console.log('Data submitted:', data);
    output.textContent = `You submitted: ${data}`;
    form.reset();
})