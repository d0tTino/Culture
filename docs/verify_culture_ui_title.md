# Verifying Culture UI Title

The production build should include the updated title `Culture UI` in `dist/index.html`.

After running `npx vite build` in `culture-ui`, the generated `dist/index.html` contains the expected title:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Culture UI</title>
    <!-- rest of file -->
```

```
