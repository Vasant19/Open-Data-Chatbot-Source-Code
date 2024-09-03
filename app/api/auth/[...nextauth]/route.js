import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";

const handler = NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
  ],
  callbacks: {
    async signIn({ user, account, profile, email, credentials }) {
      console.log("Sign in callback:", { user, account, profile, email });
      return true;
    },
    async session({ session, user, token }) {
      console.log("Session callback:", { session, user, token });
      return session;
    },
  },
  debug: true, // Enable debug messages
});

export { handler as GET, handler as POST };