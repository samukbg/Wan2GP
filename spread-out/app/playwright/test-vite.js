const { createServer } = require('vite');
async function test() {
  const server = await createServer({ server: { port: 3000 } });
  await server.listen();
  console.log(server.resolvedUrls.local[0]);
  server.close();
}
test();
