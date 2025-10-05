import type { Actions } from '@sveltejs/kit';
import { fail } from '@sveltejs/kit';
import { CLASSIFIER_SERVER } from '$env/static/private';

export const actions = {
	submit: async ({ request }) => {
		// Get the form data
		const formData = await request.formData();

		const fileUrl = formData.get('fileURL');
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
			classifierFormData.append('fileURL', fileUrl as string);
		}

		try {
			// Send to classifier server
			const response = await fetch( CLASSIFIER_SERVER + `/evaluate/`, {
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
// 			console.log('Classifier result:', result);

			return result;
		} catch (err) {

// 			// testing
// 			return {
// 				success: true,
// 				result: "success",
// 				segments: [
// 					{
// 						startTime: {
// 							minute: 0,
// 							second: 0
// 						},
// 						endTime: {
// 							minute: 0,
// 							second: 30
// 						},
// 						text: "I hate black people.",
// 						extreme: 0.99
// 					},
// 					{
// 						startTime: {
// 							minute: 1,
// 							second: 0
// 						},
// 						endTime: {
// 							minute: 1,
// 							second: 4
// 						},
// 						text: "I am cool,",
// 						extreme: 0.01
// 					},
// 										{
// 						startTime: {
// 							minute: 1,
// 							second: 30
// 						},
// 						endTime: {
// 							minute: 1,
// 							second: 40
// 						},
// 						text: "I kinda hate others.",
// 						extreme: 0.61
// 					}
// 				]
// 			}

			console.error('Failed to connect to classifier server:', err);
			return fail(500, {
				error: 'Failed to connect to classifier server'
			});
		}
	}
} satisfies Actions;
