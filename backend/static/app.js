// app.js
window.addEventListener("load", async () => {
    try {
        // Check if Clerk script loaded
        if (!window.Clerk) {
            throw new Error("Clerk script not loaded. Check your internet connection or the script URL.");
        }

        const scriptTag = document.querySelector('script[data-clerk-publishable-key]');
        const key = scriptTag ? scriptTag.getAttribute('data-clerk-publishable-key') : null;

        if (!key || key === 'None' || key === '') {
            throw new Error("Clerk Publishable Key is missing. Please check your .env file.");
        }

        // Wait for Clerk to load
        await Clerk.load();

        if (!Clerk.user) {
            // Mount Sign In component
            const appDiv = document.getElementById("app");
            Clerk.mountSignIn(appDiv);
        } else {
            // Show Dashboard if logged in
            showDashboard();
        }
    } catch (err) {
        console.error("Clerk Init Error:", err);
        document.getElementById("app").innerHTML = `
      <div style="color: red; padding: 20px;">
        <h3>Error Loading App</h3>
        <p>${err.message}</p>
        <p><strong>Make sure CLERK_PUBLISHABLE_KEY is set in your .env file.</strong></p>
      </div>
    `;
    }
});

async function showDashboard() {
    window.location.href = "/home";
}

async function callAgent() {
    try {
        // Get the session token
        const token = await Clerk.session.getToken();

        // Call the Flask backend
        const res = await fetch("/agent", { // Relative path since we are on same origin now
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: "Hello agent"
            })
        });

        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        alert("Backend Reply: " + data.reply);
    } catch (error) {
        console.error("Error calling agent:", error);
        alert("Error calling agent. Check console for details.");
    }
}
