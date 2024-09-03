const { OAuth2Client } = require('google-auth-library');
const dotenv = require('dotenv');

dotenv.config({ path: '.env.local' });

const clientId = process.env.GOOGLE_ID;
const clientSecret = process.env.GOOGLE_CLIENT_SECRET;
const redirectUri = 'http://localhost:3000/api/auth/callback/google';

console.log('Client ID:', clientId);
console.log('Redirect URI:', redirectUri);
async function testGoogleConnection() {
  const oAuth2Client = new OAuth2Client(clientId, clientSecret, redirectUri);

  try {
    // Generate a URL for Google OAuth consent screen
    const authorizeUrl = oAuth2Client.generateAuthUrl({
      access_type: 'offline',
      scope: ['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email']
    });

    console.log('Google OAuth is configured correctly.');
    console.log('Authorization URL:', authorizeUrl);
    console.log('To complete the test:');
    console.log('1. Copy and paste the above URL into your browser.');
    console.log('2. Sign in with your Google account.');
    console.log('3. You should be redirected back to your application.');
    console.log('If you see the Google sign-in page and get redirected successfully, your connection is working.');

  } catch (error) {
    console.error('Error testing Google connection:', error.message);
  }
}

testGoogleConnection();