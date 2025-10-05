<script lang="ts">
  import { timestampToString } from "$lib/utils";
  import type { TimeStamp } from "$lib/utils";

  type Segment = {
    startTime: TimeStamp;
    endTime: TimeStamp;
    text: string;
    extreme: number; // [0,1]
    hateTarget?: string;
    hateTargetConfidence?: number;
  };

  type FormResult = {
    success?: boolean;
    result?: string;
    segments?: Segment[];
    error?: string;
    isExtremist?: boolean;
    full_text_classification?: any;
    statistics?: any;
  };

  let uploadedResults: FormResult | null = null;
  let error = "";
  let originalFileName = "";

  async function handleJsonUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    
    if (!file) return;
    
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      
      // Validate that it's a valid result format
      if (json.success !== undefined && json.segments !== undefined) {
        uploadedResults = json;
        // Extract original filename from the uploaded JSON filename
        originalFileName = file.name.replace(/-result\.json$/, "") || "uploaded";
        error = "";
      } else {
        error = "Invalid JSON format. Expected result format with 'success' and 'segments' fields.";
        uploadedResults = null;
      }
    } catch (e) {
      error = `Failed to parse JSON file: ${e instanceof Error ? e.message : 'Unknown error'}`;
      uploadedResults = null;
    }
    
    // Clear the input
    input.value = "";
  }

  function clearResults() {
    uploadedResults = null;
    originalFileName = "";
    error = "";
  }
</script>

<h1>View Previous Results</h1>
<p><a href="/">← Back to Analysis</a></p>
<hr/>

<hgroup>
  <h3>Upload Results JSON</h3>
  <p>Load a previously saved JSON result file to view the analysis.</p>
</hgroup>

<input 
  type="file" 
  accept="application/json,.json" 
  on:change={handleJsonUpload}
  style="margin-bottom: 1rem;"
/>

{#if uploadedResults}
  <button on:click={clearResults} style="margin-bottom: 1rem;">
    Clear Results
  </button>
{/if}

{#if error}
  <article style="border-style: solid; border-color: var(--pico-form-element-invalid-border-color); padding: 1rem;">
    <h2>Error</h2>
    <p>{error}</p>
  </article>
{/if}

{#if uploadedResults?.success && uploadedResults?.segments}
  <article>
    <h2>Analysis Results</h2>
    {#if originalFileName}
      <p><strong>File:</strong> {originalFileName}</p>
    {/if}
    
    <h3>Summary</h3>
    <pre>{JSON.stringify(uploadedResults.result, null, 2)}</pre>

    {#if uploadedResults.statistics}
      <details>
        <summary>Statistics</summary>
        <pre>{JSON.stringify(uploadedResults.statistics, null, 2)}</pre>
      </details>
    {/if}

    {#if uploadedResults.full_text_classification}
      <details>
        <summary>Full Text Classification</summary>
        <pre>{JSON.stringify(uploadedResults.full_text_classification, null, 2)}</pre>
      </details>
    {/if}

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
      {#each uploadedResults.segments as segment, i (i)}
        {@const tooltipText = `${timestampToString(segment.startTime)}-${timestampToString(segment.endTime)}: ${(segment.extreme*100).toFixed(1)}%${segment.hateTarget ? `\nTarget: ${segment.hateTarget} (${((segment.hateTargetConfidence || 0)*100).toFixed(1)}%)` : ''}`}
        <span 
          data-tooltip={tooltipText}
          style={(segment.extreme > 0.5 ? `background-color: rgba(255, 0, 0, ${segment.extreme * 0.5})` : '') + "; border-bottom: 0px;"}
        >{segment.text}</span>{' '}
      {/each}
      {#if uploadedResults.segments.length === 0}
      <em>No text found in the recording</em>
      {/if}
    </div>
  </article>
{/if}
