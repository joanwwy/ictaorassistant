import { defineConfig } from '@prisma/config';

export default defineConfig({
  // This tells Prisma Migrate where to find your database during deployment
  migration: {
    url: process.env.DATABASE_URL,
  },
});