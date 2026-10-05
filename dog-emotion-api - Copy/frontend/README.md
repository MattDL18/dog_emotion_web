# frontend/

Drop the front-end files here.

The API server expects them to be served separately (or by a static file host like Vercel/Netlify/GitHub Pages).  
During local development you can open the HTML file directly in the browser — the CORS policy is open (`*`).

## Connecting to the backend

Point your fetch calls at the server address:

```js
const API = "http://localhost:8000";   // change to your tunnel / hosted URL for demo

async function predict(file, model = "hybrid") {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);          // "single" or "hybrid"
  const res = await fetch(`${API}/predict`, { method: "POST", body: form });
  return res.json();
}
```

## Model selector (recommended)

Add a `<select>` so the professor can switch between pipelines live:

```html
<select id="modelSelect">
  <option value="hybrid">Hybrid (two-stage)</option>
  <option value="single">Single (one-stage)</option>
</select>
```

Then read `document.getElementById("modelSelect").value` when building the `FormData`.
