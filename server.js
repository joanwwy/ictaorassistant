// Needed for dotenv
require("dotenv").config();

// Needed for Express
var express = require('express');
var app = express();
var multer = require('multer');
var upload = multer({ dest: 'uploads/' });

// Needed for form-data and fetch
const FormData = require('form-data');
const fetch = require('node-fetch');
const fs = require('fs');

// Needed for EJS
app.set('view engine', 'ejs');

// Needed for public directory
app.use(express.static(__dirname + '/public'));

// Needed for parsing form data
app.use(express.json());       
app.use(express.urlencoded({extended: true}));

// Needed for Prisma to connect to database
const { Pool } = require('pg');
const { PrismaPg } = require('@prisma/adapter-pg');
const { PrismaClient } = require('@prisma/client');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const adapter = new PrismaPg(pool);
const prisma = new PrismaClient({ adapter });

// Main landing page
app.get('/', function(req, res) {
    res.render('pages/home', { result: null, error: null });
});

// About landing page
app.get('/about', function(req, res) {
    res.render('pages/about');
});

// Handle AI input submission
app.post('/generate', upload.single('attachment'), async function(req, res) {
    try {
        const { userInput } = req.body;
        const file = req.file;

        if (!userInput) {
            return res.render('pages/home', { error: 'Please provide an input.', result: null });
        }

        if (!file) {
            return res.render('pages/home', { error: 'Please attach a file.', result: null });
        }

        // Build form data to send to Python backend
        const formData = new FormData();
        formData.append('query', userInput);
        formData.append('file', fs.createReadStream(file.path), file.originalname);

        // Call Python backend
        const response = await fetch(process.env.PYTHON_BACKEND_URL + '/process', {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
        });

        const data = await response.json();

        // Clean up uploaded file after processing
        fs.unlinkSync(file.path);

        res.render('pages/home', { result: data.result, error: null });
    } catch (error) {
        console.log(error);
        res.render('pages/home', { error: 'Something went wrong.', result: null });
    }
});

// Tells the app which port to run on
const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0',() => {
    console.log(`Server running on port ${PORT}`);
});