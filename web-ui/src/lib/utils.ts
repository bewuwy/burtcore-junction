export function validateUrl(url: string, allowEmpty: boolean = false): { valid: boolean; error: string } {
	if (!url || url.trim() === '') {
		if (allowEmpty) {
			return {
				valid: true,
				error: ''
			};
		}
		return {
			valid: false,
			error: 'File URL is required'
		};
	}

	try {
		const urlObj = new URL(url);
		if (urlObj.protocol !== 'http:' && urlObj.protocol !== 'https:') {
			return {
				valid: false,
				error: 'File URL must use http or https protocol'
			};
		}
		return {
			valid: true,
			error: ''
		};
	} catch {
		return {
			valid: false,
			error: 'Invalid URL format'
		};
	}
}
