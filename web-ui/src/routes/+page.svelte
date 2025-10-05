<script lang="ts">
  import { enhance } from "$app/forms";
  import { validateUrl, timestampToString } from "$lib/utils";
  import type { TimeStamp } from "$lib/utils";

  type Segment = {
    startTime: TimeStamp;
    endTime: TimeStamp;
    text: string;
    extreme: number; // [0,1]
    hateTarget?: string;
    hateTargetConfidence?: number;
    intonation?: {
      emotion: string;
      emotion_score: number;
    }
  };

  type FormResult = {
    success?: boolean;
    result?: string;
    segments?: Segment[];
    error?: string;
  };

  export let form: FormResult | null = null;

  let loading = false;
  let error = "";
  let originalFileName = "";

  function downloadResults() {
    if (!form) return;
    
    // Create the filename based on original file name
    const baseFileName = originalFileName.replace(/\.[^/.]+$/, "") || "result";
    const fileName = `${baseFileName}-result.json`;
    
    // Create the JSON content
    const jsonContent = JSON.stringify(form, null, 2);
    
    // Create blob and download
    const blob = new Blob([jsonContent], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const emotionMap: { [key: string]: string } = {
    hap: 'Happy',
    ang: 'Angry',
    neu: 'Neutral',
    sad: 'Sad',
    dis: 'Disgust',
    fea: 'Fear',
    sur: 'Surprise',
  };

  function getFullEmotionName(shortName: string): string {
    return emotionMap[shortName.toLowerCase()] || shortName;
  }
</script>

<h1>BurtCore Extremist Speech detector</h1>
<p><a href="/view-results">View Previous Results →</a></p>
<hr/>

<form
  method="POST"
  action="?/submit"
  enctype="multipart/form-data"
  use:enhance={({ formData }) => {
    const fileUrl = formData.get('fileURL') as string;
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
    
    // Store the original file name for later use
    if (hasFile) {
      originalFileName = file.name;
    } else if (fileUrl) {
      // Extract filename from URL
      const urlPath = fileUrl.split('?')[0];
      originalFileName = urlPath.split('/').pop() || "video";
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
  <p>You can upload a file or input its URL</p>
</hgroup>

  <fieldset class="grid">
    <label>
      File
      <input type="file" name="file" accept="audio/*,video/*" />
    </label>

    <label>
      File URL
      <input
        type="text"
        name="fileURL"
        placeholder="https://hate.com/speech.mp4"
      />
    </label>
  </fieldset>

  <input type="submit" value="Evaluate" disabled={loading} />
</form>

{#if loading}
  <p aria-busy="true">Processing... Please wait.</p>
{/if}

{#if error && !form?.success}
  <article style="border-style: solid; border-color: var(--pico-form-element-invalid-border-color); padding: 1rem;">
    <h2>Error</h2>
    <p>{error}</p>
  </article>
{/if}

{#if !error && form?.error && !form?.success}
  <article style="border-style: solid; border-color: var(--pico-form-element-invalid-border-color); padding: 1rem;">
    <h2>Server Error</h2>
    <p>{form.error}</p>
  </article>
{/if}

{#if form?.success && form?.segments}
  <article>
    <h2>Classification Result</h2>
    <pre>{JSON.stringify(form.result, null, 2)}</pre>

    <a href={"#"} on:click={downloadResults}>
      Download Results as JSON
    </a>

    <hgroup>
      <h4>Transcript</h4>
      <p>Hover a segment to see its timestamp and probability of it being extremist. Suspicious segments are colored red. Hate targets are shown when detected.</p>
    </hgroup>

    <style>
      span:hover {
        border-bottom: 2px dotted !important;
      }
    </style>

    <div>
      {#each form?.segments as segment, i (i)}
        {@const intonation = segment.intonation}
        {@const fullEmotion = intonation ? getFullEmotionName(intonation.emotion) : ''}
        {@const tooltipText = `${timestampToString(segment.startTime)}-${timestampToString(segment.endTime)}: ${(segment.extreme*100).toFixed(1)}%${segment.hateTarget ? `\nTarget: ${segment.hateTarget} (${((segment.hateTargetConfidence || 0)*100).toFixed(1)}%)` : ''}${intonation ? `\nEmotion: ${fullEmotion} (${(intonation.emotion_score*100).toFixed(1)}%)` : ''}`}
        <span 
          data-tooltip={tooltipText}
          style={(segment.extreme > 0.5 ? `background-color: rgba(255, 0, 0, ${segment.extreme * 0.5})` : '') + "; border-bottom: 0px;"}
        >{segment.text}</span>{' '}
      {/each}
      {#if form?.segments.length === 0}
      <em>No text found in the recording</em>
      {/if}
    </div>
    </article>
{/if}
