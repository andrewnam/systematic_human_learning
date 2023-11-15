/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// const taskURL = "http://localhost:3000/";
const taskURL = "http://52.53.212.49:5000/";
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);
var pages = [
	"stage.html"
];

psiTurk.preloadPages(pages);

var Experiment = function() {

	var onComplete = Math.random().toString(36).substring(2);
	psiTurk.showPage('stage.html');
	$('#iframe').attr('src', taskURL + new URL(window.location.href).search + "&onComplete=" + onComplete);

	window.addEventListener('message', function(event){

		// normally there would be a security check here on event.origin (see the MDN link above), but meh.
		if (event.data) {
			if (typeof event.data === 'string') {
				if (event.data == onComplete) {
					psiTurk.completeHIT();
				}
			}
		}
	})
};


// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
	currentview = new Experiment();
});