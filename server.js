// Needed for dotenv
require("dotenv").config();

// Needed for Express
var express = require('express');
var app = express();
var multer = require('multer');
var upload = multer({ dest: 'uploads/' });

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
        const file = req.file; // will be undefined if no file uploaded

        if (!userInput) {
            return res.render('pages/home', { error: 'Please provide an input.', result: null });
        }

        // Save the submission to the database
        const submission = await prisma.submission.create({
            data: {
                userInput: userInput,
                fileName: file ? file.originalname : null,
                filePath: file ? file.path : null, // store S3 URL here instead if using cloud storage
            },
        });

        // Call your AI API here, passing userInput and/or file contents
        // const result = await callAI(userInput, file);
        const result = `Submission saved with ID: ${submission.id}`; // placeholder

        res.render('pages/home', { result: result, error: null });
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