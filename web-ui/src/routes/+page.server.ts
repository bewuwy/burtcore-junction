import type { Actions } from '@sveltejs/kit';
import { fail } from '@sveltejs/kit';
import { CLASSIFIER_SERVER } from '$env/static/private';

export const actions = {
	submit: async ({ request }) => {
		// Get the form data
		const formData = await request.formData();

		const fileUrl = formData.get('file_url');
		const file = formData.get('file');

		console.log('File URL:', fileUrl);
		console.log('File:', file);

		// Prepare data to send to classifier server
		const classifierFormData = new FormData();
		
		if (file && file instanceof File && file.size > 0) {
			// If file is uploaded, send the file
			classifierFormData.append('file', file);
		} else if (fileUrl) {
			// If file URL is provided, send the URL
			classifierFormData.append('file_url', fileUrl as string);
		}

		try {
			// Send to classifier server
			const response = await fetch(`${CLASSIFIER_SERVER}/evaluate`, {
				method: 'POST',
				body: classifierFormData
			});

			if (!response.ok) {
				const errorText = await response.text();
				console.error('Classifier server error:', errorText);
				return fail(500, {
					error: `Classifier server error: ${response.status} ${response.statusText}`
				});
			}

			const result = await response.json();
			console.log('Classifier result:', result);

			return {
				success: true,
				result
			};
		} catch (err) {
			console.error('Failed to connect to classifier server:', err);
			return fail(500, {
				error: 'Failed to connect to classifier server'
			});
		}
	}
} satisfies Actions;
