(function() {
    var resources = [];
    var seenUrls = new Set();

    function isAboveTheFold(element) {
        var rect = element.getBoundingClientRect();
        if (!rect.bottom){
            return false
        }
        return rect.bottom <= window.innerHeight;
    }

    function getBackgroundImageUrl(element) {
        var style = window.getComputedStyle(element);
        var backgroundImage = style.getPropertyValue('background-image');
        if (backgroundImage !== 'none') {
            return backgroundImage.replace(/url\(['"]?(.*?)['"]?\)/i, '$1');
        }
        return null;
    }

    function extractUrlsFromCssText(cssText) {
        const urlRegex = /url\(['"]?(.*?)['"]?\)/gi;
        const urls = [];
        let match;
        
        while ((match = urlRegex.exec(cssText)) !== null) {
            if (match[1]) {
                urls.push(match[1]);
            }
        }
        
        return urls;
    }

    function addResource(resource) {
        if (resource.src) {
            try {
                let normalizedUrl = new URL(resource.src, window.location.href).href;
                
                if (!seenUrls.has(normalizedUrl)) {
                    seenUrls.add(normalizedUrl);
                    resource.src = normalizedUrl;
                    resources.push(resource);
                } else if (resource.isAboveTheFold) {
                    const existingResource = resources.find(r => r.src === normalizedUrl);
                    if (existingResource) {
                        existingResource.isAboveTheFold = true;
                        if (existingResource.width && existingResource.height && resource.width && resource.height){
                            const existingSize = existingResource.width * existingResource.height;
                            const newSize = resource.width * resource.height;
                            if (newSize > existingSize){
                                existingResource.width = resource.width;
                                existingResource.height = resource.height;
                            }
                        }
                    }
                }
            } catch (e) {
                console.warn('Invalid URL:', resource.src);
            }
        } else {
            resources.push(resource);
        }
    }

    function processCSSImages() {
        Array.from(document.styleSheets).forEach(function(stylesheet) {
            try {
                if (stylesheet.href && !stylesheet.cssRules) {
                    return;
                }

                Array.from(stylesheet.cssRules || []).forEach(function(rule) {
                    if (rule instanceof CSSStyleRule) {
                        const backgroundImages = extractUrlsFromCssText(rule.style.cssText);
                        backgroundImages.forEach(url => {
                            addResource({
                                type: 'image',
                                img_type: 'css',
                                src: url,
                            });
                        });
                    }
                    else if (rule instanceof CSSImportRule && rule.styleSheet) {
                        processCSSStyleSheet(rule.styleSheet);
                    }
                    else if (rule instanceof CSSMediaRule) {
                        Array.from(rule.cssRules).forEach(mediaRule => {
                            if (mediaRule instanceof CSSStyleRule) {
                                const backgroundImages = extractUrlsFromCssText(mediaRule.style.cssText);
                                backgroundImages.forEach(url => {
                                    addResource({
                                        type: 'image',
                                        img_type: 'css',
                                        src: url,
                                    });
                                });
                            }
                        });
                    }
                });
            } catch (e) {
                console.warn('Could not process stylesheet:', stylesheet.href, e);
            }
        });
    }

    function processBackgroundImages(element) {
        var backgroundImageUrl = getBackgroundImageUrl(element);
        if (backgroundImageUrl) {
            if (element.getBoundingClientRect) {
                var elRect = element.getBoundingClientRect();
                const style = window.getComputedStyle(element);
                addResource({
                    type: 'image',
                    img_type: 'background',
                    src: backgroundImageUrl,
                    isAboveTheFold: isAboveTheFold(element),
                    width: Math.round(elRect.width),
                    height: Math.round(elRect.height),
                    bottom: Math.round(elRect.bottom),
                    winheight: Math.round(window.innerHeight),
                    display: style.display,
                    visibility: style.visibility 
                });
            }
        }

        Array.from(element.children).forEach(processBackgroundImages);
    }

    Array.from(document.scripts).forEach(function(script) {
        if (script.src) {
            addResource({
                type: 'js',
                src: script.src,
                async: script.async,
                defer: script.defer,
            });
        }
    });

    Array.from(document.styleSheets).forEach(function(stylesheet) {
        if (stylesheet.href) {
            addResource({
                type: 'css',
                src: stylesheet.href,
            });
        }
    });

    function processImageElement(img) {
        var elRect = img.getBoundingClientRect();
        const style = window.getComputedStyle(img);
        
        // Get both natural and rendered dimensions
        const naturalWidth = img.naturalWidth || img.width || 0;
        const naturalHeight = img.naturalHeight || img.height || 0;
        const renderedWidth = Math.round(elRect.width) || 0;
        const renderedHeight = Math.round(elRect.height) || 0;
    
        // Use rendered dimensions if available, fall back to natural dimensions
        const finalWidth = renderedWidth || naturalWidth;
        const finalHeight = renderedHeight || naturalHeight;
    
        addResource({
            type: 'image',
            img_type: 'image',
            src: img.currentSrc || img.src,
            width: finalWidth,
            height: finalHeight,
            naturalWidth: naturalWidth,  // might be useful to keep track of both
            naturalHeight: naturalHeight,
            loading: img.loading,
            isAboveTheFold: isAboveTheFold(img),
            bottom: Math.round(elRect.bottom),
            winheight: Math.round(window.innerHeight),
            display: style.display,
            visibility: style.visibility 
        });
    }

    function getBackgroundImageUrl(style) {
        const bgImage = style['background-image'];
        if (bgImage && bgImage !== 'none') {
            const matches = bgImage.match(/url\(['"]?(.*?)['"]?\)/);
            if (matches && matches[1]) {
                return matches[1].replace(/[''"]/g, '');
            }
        }
        return null;
    }

    function getCSSImageSizes(){
        var elements = document.getElementsByTagName('*');
        var re = /url\(.*(http.*)\)/ig;
        for (var i = 0; i < elements.length; i++) {
            var el = elements[i];
            var style = window.getComputedStyle(el);
            

            if (style['background-image']) {
                var url = getBackgroundImageUrl(style)
                if (url){
                    re.lastIndex = 0;
                    var matches = re.exec(style['background-image']);
                    if (matches && matches.length > 1)
                    var elRect = el.getBoundingClientRect()
                    if (elRect) {
                        addResource({
                            type: 'image',
                            img_type: 'css',
                            src: url,
                            width: elRect.width,
                            height: elRect.height,
                            isAboveTheFold: isAboveTheFold(el),
                            bottom: Math.round(elRect.bottom),
                            winheight: Math.round(window.innerHeight),
                            display: style.display,
                            visibility: style.visibility 
                        })
                    }
                }
                
            }
        }
    }

    Array.from(document.getElementsByTagName('img')).forEach(processImageElement);
    Array.from(document.getElementsByTagName('picture')).forEach(function(picture) {
        var img = picture.querySelector('img');
        if (img) processImageElement(img);
    });

    Array.from(document.getElementsByTagName('svg')).forEach(function(svg) {
        var elRect = svg.getBoundingClientRect();
        var src = svg.getAttribute('src') || svg.getAttribute('data');
        const style = window.getComputedStyle(svg);
        if (src) {  // Only add if there's a source URL (skips inline SVGs)
            addResource({
                type: 'image',
                img_type: 'svg',
                src: src,
                isAboveTheFold: isAboveTheFold(svg),
                width: Math.round(elRect.width),
                height: Math.round(elRect.height),
                bottom: Math.round(elRect.bottom),
                winheight: Math.round(window.innerHeight),
                display: style.display,
                visibility: style.visibility 
            });
        }
    });

    processBackgroundImages(document.body);
    getCSSImageSizes();
    processCSSImages();

    function getPseudoElementBackgroundImage(element, pseudo) {
        var style = window.getComputedStyle(element, pseudo);
        var backgroundImage = style.getPropertyValue('background-image');
        if (backgroundImage !== 'none') {
            var elRect = element.getBoundingClientRect();
            addResource({
                type: 'image',
                img_type: 'pseudo',
                src: backgroundImage.replace(/url\(['"]?(.*?)['"]?\)/i, '$1'),
                isAboveTheFold: isAboveTheFold(element),
                width: Math.round(elRect.width),
                height: Math.round(elRect.height),
                bottom: Math.round(elRect.bottom),
                winheight: Math.round(window.innerHeight),
                display: style.display,
                visibility: style.visibility 
            });
        }
    }

    function processPseudoElements(element) {
        getPseudoElementBackgroundImage(element, ':before');
        getPseudoElementBackgroundImage(element, ':after');
        Array.from(element.children).forEach(processPseudoElements);
    }

    processPseudoElements(document.body);

    return resources;
})();