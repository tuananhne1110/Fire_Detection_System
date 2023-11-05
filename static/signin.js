function scrollToSignInMiddle() {
    const signInContainer = document.querySelector('.sign-in-container');
    
    if (signInContainer) {
 
        const middleOfContainer = signInContainer.getBoundingClientRect().top + (signInContainer.clientHeight - 650);

        window.scrollTo({
            top: middleOfContainer,
            behavior: 'auto'
        });
    }
}

window.addEventListener('load', scrollToSignInMiddle);