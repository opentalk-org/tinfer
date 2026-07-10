// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Tinfer',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com' }],
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{
					label: 'Getting started',
					items: [
						{ slug: 'introduction' },
						{ slug: 'quickstart' },
					],
				},
				{
					label: 'Concepts',
					items: [
						{ slug: 'concepts/streaming' },
						{ slug: 'concepts/alignments' },
						{ slug: 'concepts/parameters' },
						{ slug: 'concepts/errors' },
					],
				},
				{
					label: 'Server',
					items: [
						{ slug: 'server/overview' },
						{ slug: 'server/grpc' },
						{ slug: 'server/websocket' },
					],
				},
				{
					label: 'API reference',
					items: [
						{ slug: 'api/streaming-tts' },
						{ slug: 'api/async-streaming-tts' },
						{ slug: 'api/tts-stream' },
						{ slug: 'api/audio-chunk' },
						{ slug: 'api/alignment' },
						{ slug: 'api/streaming-tts-config' },
						{ slug: 'api/styletts2-params' },
						{ slug: 'api/audio-format' },
					],
				},
			],
		}),
	],
});
