<script lang="ts">
  import { enhance } from "$app/forms";
  import { validateUrl } from "$lib/utils";

  type FormResult = {
    success?: boolean;
    result?: any;
    error?: string;
  };

  export let form: FormResult | null = null;

  let loading = false;
  let error = "";
</script>

<h1>Welcome to ExtremeDetector5000(TM)</h1>
<hr/>

<form
  method="POST"
  action="?/submit"
  enctype="multipart/form-data"
  use:enhance={({ formData }) => {
    const fileUrl = formData.get('file_url') as string;
    const file = formData.get('file') as File;
    
    const hasFile = file && file.size > 0;
    const validation = validateUrl(fileUrl, hasFile);    

    if (!hasFile && (!fileUrl || fileUrl.trim() === '')) {
      error = 'Please provide either a file or a file URL';
      return async () => {};
    }

    if (!validation.valid) {
      error = validation.error;
      return async () => {}; // Cancel the submission
    }
    
    console.log("Form submission started");
    loading = true;
    error = "";
    return async ({ update }) => {
      console.log("Form submission completed");
      await update();
      loading = false;
    };
  }}
>

<hgroup>

  <h4>Audio/Video file to check for extremist speech:</h4>
  <p>You can upload or a file or input its URL</p>
</hgroup>

  <fieldset class="grid">
    <label>
      File
      <input type="file" name="file" />
    </label>

    <label>
      File URL
      <input
        type="text"
        name="file_url"
        placeholder="https://hate.com/speech.mp4"
      />
    </label>
  </fieldset>

  <input type="submit" value="Evaluate" disabled={loading} />
</form>

{#if loading}
  <p aria-busy="true">Processing... Please wait.</p>
{/if}

{#if error}
  <article style="background-color: var(--pico-form-element-invalid-border-color); padding: 1rem;">
    <h2>Error</h2>
    <p>{error}</p>
  </article>
{/if}

{#if form?.error}
  <article style="background-color: var(--pico-form-element-invalid-border-color); padding: 1rem;">
    <h2>Server Error</h2>
    <p>{form.error}</p>
  </article>
{/if}

{#if form?.success && form?.result}
  <article>
    <h2>Classification Result</h2>
    <pre>{JSON.stringify(form.result, null, 2)}</pre>
  </article>
{/if}
