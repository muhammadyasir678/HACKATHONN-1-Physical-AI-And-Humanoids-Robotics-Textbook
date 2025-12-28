import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Textbook sidebar with structured modules and chapters
  textbookSidebar: [
    {
      type: 'doc',
      id: 'introduction/intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-robotic-nervous-system/chapter-1',
        'module-1-robotic-nervous-system/chapter-2',
        'module-1-robotic-nervous-system/chapter-3',
        'module-1-robotic-nervous-system/chapter-4',
        'module-1-robotic-nervous-system/chapter-5',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/chapter-1',
        'module-2-digital-twin/chapter-2',
        'module-2-digital-twin/chapter-3',
        'module-2-digital-twin/chapter-4',
        'module-2-digital-twin/chapter-5',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-robot-brain/chapter-1',
        'module-3-ai-robot-brain/chapter-2',
        'module-3-ai-robot-brain/chapter-3',
        'module-3-ai-robot-brain/chapter-4',
        'module-3-ai-robot-brain/chapter-5',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vision-language-action/chapter-1',
        'module-4-vision-language-action/chapter-2',
        'module-4-vision-language-action/chapter-3',
        'module-4-vision-language-action/chapter-4',
        'module-4-vision-language-action/chapter-5',
      ],
    },
  ],
};

export default sidebars;
