function scrollToSignInMiddle() {
    const signInContainer = document.querySelector('.sign-in-container');
    
    if (signInContainer) {
 
        const middleOfContainer = signInContainer.getBoundingClientRect().top + (signInContainer.clientHeight / 20);

        window.scrollTo({
            top: middleOfContainer,
            behavior: 'smooth'
        });
    }
}

window.addEventListener('load', scrollToSignInMiddle);