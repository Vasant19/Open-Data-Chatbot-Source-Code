const mysql = require('mysql2/promise');
const dotenv = require('dotenv');

dotenv.config({ path: '.env.local' });

async function testConnection() {
  const connection = await mysql.createConnection({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  });

  try {
    // Test the connection
    await connection.connect();
    console.log('Successfully connected to MySQL database');

    // Perform a simple query
    const [rows] = await connection.execute('SELECT 1 + 1 AS result');
    console.log('Query result:', rows[0].result);

    // List all tables in the database
    const [tables] = await connection.execute('SHOW TABLES');
    console.log('Tables in the database:');
    tables.forEach(table => {
      console.log(table[`Tables_in_${process.env.DB_NAME}`]);
    });

  } catch (error) {
    console.error('Error connecting to the database:', error);
  } finally {
    await connection.end();
  }
}

testConnection();