
import { spawn } from "child_process";

console.log("Starting Streamlit application...");

const streamlit = spawn("streamlit", [
  "run",
  "app.py",
  "--server.port", "5000",
  "--server.address", "0.0.0.0",
  "--server.headless", "true",
  "--global.developmentMode", "false"
], { stdio: "inherit" });

streamlit.on("close", (code) => {
  console.log(`Streamlit exited with code ${code}`);
  process.exit(code ?? 0);
});

streamlit.on("error", (err) => {
  console.error("Failed to start Streamlit:", err);
});
