const { PrismaClient } = require('@prisma/client')
const prisma = new PrismaClient({
  // This completely bypasses prisma.config.ts and forces the engine to read the string directly
  __internal: {
    configOverride: (config) => {
      config.inlineSchema = config.inlineSchema.replace(
        /url\s*=\s*env\("DATABASE_URL"\)/,
        `url = "${process.env.DATABASE_URL}"`
      );
      return config;
    }
  }
});