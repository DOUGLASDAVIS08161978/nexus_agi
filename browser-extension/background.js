/**
 * Background service worker for Frequency Medicine AI Injector
 */

// Initialize storage
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ injectionCount: 0 });
  console.log('[Frequency Medicine Injector] Extension installed');
});

// Listen for tab updates to inject on supported sites
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    const supportedSites = [
      'chat.openai.com',
      'claude.ai',
      'gemini.google.com',
      'perplexity.ai',
      'huggingface.co/chat',
      'poe.com'
    ];

    const isSupported = supportedSites.some(site => tab.url.includes(site));

    if (isSupported) {
      console.log(`[Frequency Medicine Injector] Supported site detected: ${tab.url}`);
    }
  }
});
